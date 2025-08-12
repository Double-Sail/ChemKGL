import logging
import os

handlers = [
    logging.FileHandler('logs/rag.log', mode='w'),
    logging.StreamHandler()  # 打印到终端
]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logger = logging.getLogger(__name__)

import torch
from retrieval.graph_dao import GraphDao


class Extract:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        # self.relation_dict_name2ids = relation_dict_name2ids
        # self.relation_dict_id2names = relation_dict_id2names
        self._dao = GraphDao(args=args)

    def relation_prob(self, input_text, candidate_texts):
        # print("input_text", input_text)
        # Calculate probabilities for each candidate sequence
        candidate_probabilities = []
        if 'lama' in self.args.llm_model_path:
            input_token = self.tokenizer.tokenize(input_text)
        else:
            input_token = self.tokenizer.tokenize(input_text)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.model.dtype):
            for cand_text in candidate_texts:
                if self.args.llm_model_name.startswith('llama') or self.args.llm_model_name.startswith('vicuna'):
                    cand_token = self.tokenizer.tokenize(cand_text)
                else:
                    cand_token = self.tokenizer.tokenize(' ' + cand_text)
                prompt = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_token + cand_token)])

                logits = self.model(prompt.to(self.model.device)).logits

                probas = torch.softmax(logits, -1).squeeze()
                cand_token_ids = self.tokenizer.convert_tokens_to_ids(cand_token)
                valid_positions = torch.arange(
                    start=len(input_token) - 1,
                    end=len(input_token) + len(cand_token) - 1
                )

                token_probs = probas[valid_positions, torch.tensor(cand_token_ids).to(self.model.device)]
                log_probs = torch.log(token_probs.clamp(min=1e-12))  # 防止log(0)
                avg_log_prob = log_probs.mean()

                candidate_probabilities.append(torch.exp(avg_log_prob).item())
                # del prompt, logits, probas, token_probs, log_probs
                # # torch.cuda.empty_cache()
        return torch.tensor(candidate_probabilities).to(self.model.device)

    def retr_fact_KG_sole_prob(self, input_text, text, entity, other_entity):
        # 检索得到该实体对应的图节点
        node = self._dao.query_node('entity', name=entity).first()
        if node is None:
            return None, None, None, None

        # 根据节点搜索得到相邻的一跳关系与尾节点
        ename_facts, eid_facts = self._dao.query_triples_by_node(node, other_entity)

        relation_names = set()
        for ename_fact in ename_facts:
            if self.args.use_history_info:
                relation_names.add(ename_fact.split('|')[1] + '\n' + ename_fact.split('|')[2])
            else:
                relation_names.add(ename_fact.split('|')[1])
        relation_names = list(relation_names)

        if not relation_names:
            return None, None, None, None
        # 算各个关系与问题+答案，以及和答案的互信息（prob2需要写成text[len(input_text):]，因为问题不断迭代）（prob1=p(r1|qh1),prob2=p(r1|h1)）
        prob1 = self.relation_prob(text + '\n' + entity, relation_names)
        prob2 = self.relation_prob(text[len(input_text):] + '\n' + entity, relation_names)
        # return_prob = p(q,G=Gs)/p(G=Gs)
        return_prob = prob1 / prob2
        if self.args.loss == "prob_div":
            new_prob = (prob1 / prob2)
        elif self.args.loss == "prob_div_log":
            new_prob = (prob1 / prob2) * torch.log2(prob1 / prob2)

        if new_prob.shape[0] > self.args.beam_width:
            key = self.args.beam_width
        else:
            key = new_prob.shape[0]

        top_prob, index = torch.topk(new_prob, key)
        # return_top_prob = return_prob[index]
        selected_relations_dict = {relation_names[i.item()].split('\n')[0]: return_prob[i] for i in index}

        next_rel_entities_name = []
        next_rel_entities_id = []
        return_top_prob = []
        assert len(ename_facts) == len(eid_facts)
        for i in range(len(ename_facts)):
            ename_fact = ename_facts[i]
            eid_fact = eid_facts[i]
            head_name, relation_name, history, tail_name = ename_fact.split('|')
            head_id, relation_id, tail_id = eid_fact.split('|')

            if relation_name in selected_relations_dict:
                if self.args.use_reverse_retrieval:
                    next_rel_entities_name.append(relation_name + '\t' + head_name)
                    next_rel_entities_id.append(relation_id + '\t' + head_id)
                    return_top_prob.append(selected_relations_dict[relation_name].to("cpu").tolist())
                else:
                    next_rel_entities_name.append(relation_name + '\t' + tail_name)
                    next_rel_entities_id.append(relation_id + '\t' + tail_id)
                    return_top_prob.append(selected_relations_dict[relation_name].to("cpu").tolist())

        if self.args.use_reverse_retrieval:
            fact_set = [next_rel_entity.split('\t')[1] + '|' + next_rel_entity.split('\t')[0] + '|' + entity + '.' for next_rel_entity in next_rel_entities_name]
        else:
            fact_set = [entity + '|' + next_rel_entity.split('\t')[0] + '|' + next_rel_entity.split('\t')[1] + '.' for next_rel_entity in next_rel_entities_name]
        entity_name = [next_rel_entity.split('\t')[1] for next_rel_entity in next_rel_entities_name]
        entity_eid = [next_rel_entity.split('\t')[1] for next_rel_entity in next_rel_entities_id]

        return fact_set, entity_name, entity_eid, return_top_prob

    # def fact_triplet_to_sentence(self, triplet, relationstr):
    #     self._dao.query_node_by_id()

    def multi_hop_search(self, input_text, reactants, products):
        is_prune_again = False
        if self.args.tail_node_not_exist:
            # 如果目标产物在图谱中不存在，则按照贪心去处理
            # 是否需要再剪枝
            tail_node = self._dao.query_node('entity', name=products[0]).first()
            assert tail_node is None
            is_prune_again = True
            self.args.beam_width = 1

        # def multi_hop_search(self, input_text, input_entity, iteration, fact_needed):
        # ent_eid = input_entity[0]
        reactant_ent_name = reactants[0]
        other_reactant_entity = reactants[1:]

        text_list = [input_text]
        triple_list = ['']
        entity_list = [reactant_ent_name]
        # entity_eid_list = [ent_eid]
        text_prob = 1.0
        text_prob_list = [text_prob]

        # 缓存中途没有到五跳就结束的数据
        cache_text_list = []
        cache_triple_list = []
        cache_entity_list = []
        cache_text_prob_list = []

        for i in range(self.args.jump_num):

            new_text_list = []
            new_triple_list = []
            next_entity_list = []
            # next_entites_eid_list = []
            new_text_prob_list = []

            for j, text in enumerate(text_list):
                text_prob = text_prob_list[j]

                facts, next_entites, next_entites_eid, top_prob = self.retr_fact_KG_sole_prob(input_text, text, entity_list[j], other_reactant_entity)
                # 只在第一次加加速逻辑，后续的节点需要一个个判断，所以other_entity清空
                other_reactant_entity = []

                if facts is None:
                    if not is_prune_again:
                        # 中途的一些线路没有后续节点的情况，把这些语句保存，最后拼接到总体待用字符串中
                        # 这些线路会比别的线路更短一些，算是后续最优字符串挑选的依据
                        # 最终节点如果不是目标节点，直接pass
                        if entity_list[j] not in products:
                            continue
                        cache_text_list.append(text_list[j])
                        cache_triple_list.append(triple_list[j])
                        cache_entity_list.append(entity_list[j])
                        cache_text_prob_list.append(text_prob_list[j])
                    continue

                for k, fact in enumerate(facts):
                    # 只需要条件，首尾节点不需要
                    # 逆向检索，字符串拼接不一样（第一跳是和正向一样的，直接往后拼接即可）
                    if self.args.use_reverse_retrieval and i > 0:
                        new_text = text.split('\n')[0] + '\n' + fact.split('|')[1] + '\n' + '\n'.join(text.split('\n')[1:])
                    else:
                        new_text = text + '\n' + fact.split('|')[1]

                    new_text_prob = top_prob[k] * text_prob

                    triple = triple_list[j]
                    if self.args.use_reverse_retrieval:
                        new_triple = fact + '\n' + triple if i != 0 else fact
                    else:
                        new_triple = triple + '\n' + fact if i != 0 else fact

                    # 如果当前检索出的实体即是目标产物，则不再继续往后检索
                    current_entity = next_entites[k]
                    if current_entity in products and not is_prune_again:
                        cache_text_list.append(new_text)
                        cache_triple_list.append(new_triple)
                        cache_entity_list.append(next_entites[k])
                        cache_text_prob_list.append(new_text_prob)
                    else:
                        new_text_list.append(new_text)
                        new_triple_list.append(new_triple)
                        next_entity_list.append(next_entites[k])
                        # next_entites_eid_list.append(next_entites_eid[k])
                        new_text_prob_list.append(new_text_prob)

            # 必须有下一个值，否则text_list会赋值为空，导致报错
            if new_text_list:
                text_list = new_text_list
                triple_list = new_triple_list
                # print(f"{i}-th text_list", [text[len(input_text):] for text in text_list])
                entity_list = next_entity_list
                # entity_eid_list = next_entites_eid_list
                text_prob_list = new_text_prob_list
                # print(f"{i}-th text_prob_list", text_prob_list)
            else:
                # 这里是指所有的链条最后一个实体都不存在下一个实体，则可以跳出多跳的循环
                # 只要没找到下一个实体，直接可以断开循环，不可能有后续多跳实体
                break

        if not is_prune_again:
            # 检索过程中所有符合要求，包含结尾节点的，已经全部存储进入缓存列表
            if cache_text_list:
                # 补全中途缓存的数据(中途缓存的数据，一定是含有目标产物的)
                text_list = cache_text_list
                triple_list = cache_triple_list
                entity_list = cache_entity_list
                text_prob_list = cache_text_prob_list
            else:
                # 这里属于应该检索到，但是检索失败的情况，需要记录
                # 检索失败，但是并不代表直接否定全部，选出评分最高的那个输出，后续通过prune减去多余冗余信息，继续使用
                logger.info(f'\n[应该检索到但检索失败]:{text_list}')
                is_prune_again = True

        if len(text_prob_list) == 1 and text_prob_list[0] == 1.0:
            # 第一跳都不存在的情况
            return None, None, True

        text_prob = torch.tensor(text_prob_list)

        if self.args.loss == "prob_div":
            final_score = text_prob
        elif self.args.loss == "prob_div_log":
            final_score = text_prob * torch.log2(text_prob)
        _, index = torch.topk(final_score, 1)

        fact = text_list[index][len(input_text + ' '):]
        triple = triple_list[index]
        # print("======Final fact=====")
        # print(fact)

        return fact, triple, is_prune_again


class Prune:

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def ans_entroy(self, question, facts):

        input_text = ""
        for fact in facts:
            if input_text == "":
                input_text = fact
            else:
                input_text = input_text + ', ' + fact
        input_text = question + "\nGiven fact: " + input_text + ', ' + '\nAnswer:'
        # prom_text = entropy_prompt + input_text
        # print('prom_text', prom_text)

        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.args.device)
            beam0_logits = self.model(input_ids).logits[:, -1, :]

            if self.model.dtype == torch.float16:
                beam0_logits = beam0_logits.to(torch.float32)

            beam0_prob = torch.softmax(beam0_logits, dim=-1)
            ans_h = -(beam0_prob * torch.log(beam0_prob)).sum()

        return ans_h

    def facts_entropy(self, question, facts):

        num_inputs = len(facts)
        # print("num_inputs", num_inputs)
        entropy_values = [0.0] * num_inputs

        test_fact = []

        for i, fact in enumerate(facts):
            test_fact.append(fact)
            ans_h = self.ans_entroy(question, test_fact)
            entropy_values[i] = ans_h

        entropy_values = torch.tensor(entropy_values)
        entropy_values = (entropy_values - torch.min(entropy_values)) / (torch.max(entropy_values) - torch.min(entropy_values))

        return entropy_values.tolist()

    def prune_fact(self, question, facts_str, triples_str):
        pruned_facts_str, pruned_triples, entropy_val = '', [], []
        if facts_str:
            facts = facts_str.split('.\n')
            triples = triples_str.split('.\n')
            entropy_val = self.facts_entropy(question, facts)
            # print("========================")
            # print("entropy_val:", entropy_val)
            min_index = entropy_val.index(min(entropy_val))

            pruned_facts = facts[0:(min_index + 1)]  # need extra 1 to get the full chain
            pruned_facts_str = '.\n'.join(pruned_facts) + '.'

            pruned_triples = triples[0:(min_index + 1)]

        return pruned_facts_str, pruned_triples, entropy_val
