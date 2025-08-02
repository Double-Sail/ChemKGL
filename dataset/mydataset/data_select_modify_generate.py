import os
import zipfile
import json
import pandas as pd
import math
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import copy
import random
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import requests
from urllib.parse import quote
from datetime import datetime
import re

root_path = '/code/KGLLM2.0/MyRAG/dataset/mydataset'
base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def unzip_files_in_directory(input_path, output_path):
    """
    解压购买的抽取数据压缩包
    """
    # 遍历指定目录下的所有文件
    for filename in os.listdir(input_path):
        # 检查文件是否是 .zip 文件
        if filename.endswith(".zip"):
            # 构建完整的文件路径
            file_path = os.path.join(input_path, filename)
            # 打开 .zip 文件
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(output_path, filename[:-4]))
                print(f"解压完成: {filename}")


def smiles_clean(smi):
    try:
        # 将 SMILES 字符串转换为 RDKit 的分子对象
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 对分子进行标准化处理
            normizer = rdMolStandardize.Normalizer()
            mol = normizer.normalize(mol)
            # 选择分子中的最大片段
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
            # 去除分子中的电荷
            uc = rdMolStandardize.Uncharger()
            mol = uc.uncharge(mol)
            # 将处理后的分子转换回 SMILES 字符串
            # isomericSmiles=True 表示保留分子的立体化学信息
            # canonical=True 表示生成规范的 SMILES 字符串，确保相同的分子有唯一的 SMILES 表示
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            return smi
        else:
            # 如果输入的 SMILES 字符串无法转换为分子对象，返回原始smiles
            return None
    except Exception as e:
        # 捕获并打印异常信息，增加代码的健壮性
        print(f"An error occurred: {e}")
        return None


def split_string(s):
    """
    将字符串按非括号内的点号分割
    :param s: 输入字符串
    :return: 分割后的子字符串列表
    """
    stack = []
    result = []
    start_idx = 0
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}

    for i, char in enumerate(s):
        if char in bracket_pairs:
            stack.append(bracket_pairs[char])
        elif char in {')', ']', '}'}:
            if stack and stack[-1] == char:
                stack.pop()
        elif char == '.':
            if not stack:
                smiles = s[start_idx:i]
                clean_smiles = smiles_clean(smiles)
                result.append(clean_smiles)
                start_idx = i + 1

    smiles = s[start_idx:]
    clean_smiles = smiles_clean(smiles)
    result.append(clean_smiles)
    return result


def modify_multisteps(multisteps):
    """
    数据纠错，将一些乱标记的反应条件纠正过来
    """
    multisteps_modify = []
    for multistep in multisteps:
        multistep_modify = copy.deepcopy(multistep)

        stages = multistep['stages']
        stages_modify = []
        for stage in stages:
            stage_modify = copy.deepcopy(stage)

            data = stage['data']
            data_modify = copy.deepcopy(data)
            if not isinstance(data, dict):
                return None

            # 1.溶剂(有多种)
            solvent = data['Solvent']
            if isinstance(solvent, str):
                solvent = [solvent]

            # 2.试剂(有多种)
            reagent = data['Reagent']

            # 3.催化剂(有多种)
            catalyst = data['Catalyst']

            # 用于存储纠正过的条件数据
            temperature_correct = []
            ph_correct = []
            pressure_correct = []
            time_correct = []
            otherConditions_correct = []

            # 4.反应温度(有多种,或的关系)
            temperature = data['Temperature']
            if temperature:
                for a in temperature:
                    a = a.strip()
                    if 'C' not in a:
                        if a == 'rt':
                            temperature_correct.append('room temperature')
                        if a == 'Reflux':
                            temperature_correct.append(a)
                        # Temperatureset.add(str(a))
                    else:
                        temperature_correct.append(a)

            # 5.ph值
            try:
                ph = data['Ph']
            except:
                ph = data['pH']
            if ph:
                for a in ph:
                    a = a.strip()
                    if not any(char.isdigit() for char in a):
                        if a in ['Alkaline conditions', 'Acidic conditions', 'Neutral', 'alkaline condition', 'Neutral conditions']:
                            ph_correct.append(a)
                        if a == 'Reflux':
                            temperature_correct.append(a)
                        # Phset.add(str(a))
                    else:
                        ph_correct.append(a)

            # 6.反应时间
            time = data['Time']
            for a in time:
                a = a.strip()
                if not 'h' in a:
                    if 'C' in a or a == 'Heating':
                        temperature_correct.append(a)
                    if 'Torr' in a:
                        pressure_correct.append(a)
                    if 'min' in a:
                        time_correct.append(a)
                    if 'Reflux' == a:
                        temperature_correct.append(a)
                # Timeset.add(str(a))
                else:
                    time_correct.append(a)

            # 7.反应压力
            pressure = data['Pressure']
            if pressure:
                for a in pressure:
                    a = a.strip()
                    if not 'Torr' in a:
                        if a in ['Schlenk technique', 'Inert atmosphere', 'Sealed tube', 'Autoclave']:
                            otherConditions_correct.append(a)
                        if a in ['pH 7']:
                            ph_correct.append(a)
                        if a in ['0.1 MPa', 'High pressure']:
                            pressure_correct.append(a)
                        # Pressureset.add(str(a))
                    else:
                        pressure_correct.append(a)

            # 8.其余条件(惰性气氛等等)todo:统计一下其余条件的种类
            otherConditions_vacab = ['Flow reactor', 'Autoclave', 'UV-irradiation', 'mineral oil / Inert atmosphere', 'Darkness', 'Inert atmosphere', 'Sealed tube',
                                     'Microwave irradiation', 'Schlenk technique', 'Flow reactor', 'Irradiation', 'Microwave irradiation', 'Glovebox',
                                     'Supercritical conditions',
                                     'liquid N2', 'Nitrogen atmosphere', ]
            otherConditions = data['OtherConditions']
            for a in otherConditions:
                a = a.strip()
                if 'Cooling' in a or 'Inert atmosphere' in a or a in otherConditions_vacab:
                    otherConditions_correct.append(a)
                if a in ['Neutral conditions', 'Acidic conditions', 'Alkaline conditions']:
                    ph_correct.append(a)
                if a in ['High pressure']:
                    pressure_correct.append(a)
                if a in ['10 C', '0 C', '-78 C', '-10 C'] or 'Reflux' in a or 'Heating' in a:
                    temperature_correct.append(a)
                if a in ['12 h']:
                    time_correct.append(a)

            # 完善后的数据，进行字典更新
            data_modify['Solvent'] = solvent
            data_modify['Reagent'] = reagent
            data_modify['Catalyst'] = catalyst
            data_modify['Time'] = time_correct
            data_modify['Temperature'] = temperature_correct
            data_modify['Pressure'] = pressure_correct
            data_modify['Ph'] = ph_correct
            data_modify['OtherConditions'] = otherConditions_correct

            stage_modify['data'] = data_modify
            stages_modify.append(stage_modify)

        multistep_modify['stages'] = stages_modify
        multisteps_modify.append(multistep_modify)

    return multisteps_modify


def select_by_multi_step_schemes(multi_step_schemes):
    multi_step_schemes_modify = []
    for multi_step_scheme in multi_step_schemes:
        if isinstance(multi_step_scheme, str):
            # 这里为单步反应
            smiles = multi_step_scheme
        else:
            # 这里为多步反应
            smiles = multi_step_scheme['smiles']
        multi_step_scheme_modify = copy.deepcopy(multi_step_scheme)
        heads_str, tails_str = smiles.split('>>')

        assert '△' not in heads_str and '△' not in tails_str

        heads = split_string(heads_str)
        tails = split_string(tails_str)

        # smiles式子标准化错误，则跳过这个数据
        if None in heads or None in tails:
            return None

        # 对于产物在反应物中也出现的情况，予以剔除（todo：需要判断是否是催化剂）
        if any(d in tails for d in heads):
            return None

        smiles_modify = '△'.join(heads) + '>>' + '△'.join(tails)

        if isinstance(multi_step_scheme_modify, str):
            # 这里为单步反应
            multi_step_schemes_modify.append(smiles_modify)
        else:
            multi_step_scheme_modify['smiles'] = smiles_modify
            multi_step_schemes_modify.append(multi_step_scheme_modify)
    return multi_step_schemes_modify


def select_and_modify_data(input_path, output_path, start_multisteps=-1, end_multisteps=-1):
    assert start_multisteps <= end_multisteps
    """
    挑选出多步反应的数据，筛选出start_multisteps——end_multisteps之间的数据。
    对于一些基础的问题数据，予以剔除（包括一些反应步数和条件对不上，以及反应条件乱标记的情况）
    反应式字已用△隔开
    """
    need_reactions = []
    step = 0
    need_reactions_dict = {}
    for filename in os.listdir(input_path):
        if filename.endswith('zip'):
            continue
        filepath = os.path.join(input_path, filename + '/' + filename)
        for json_file_name in tqdm(os.listdir(filepath), desc=f'🚀 原始数据挑选与修改'):
            json_path = os.path.join(filepath, json_file_name)

            with open(json_path, 'r', encoding='utf-8') as json_file:
                json_dicts = json.load(json_file)
                for json_dict in json_dicts:
                    reactions = json_dict['reaction']
                    reference = json_dict['reference']

                    for reaction in reactions:
                        number_of_reaction_steps = reaction['condition']['Number_of_reaction_steps']
                        if isinstance(number_of_reaction_steps, str):
                            number_of_reaction_steps = int(eval(number_of_reaction_steps))

                        # 挑选所需多步反应的步数（值为-1则表示包含所有步数）
                        # todo:目前制作了对于多步数据的挑选，单步反应需要添加逻辑
                        if start_multisteps != -1 and end_multisteps != -1:
                            if start_multisteps > number_of_reaction_steps or end_multisteps < number_of_reaction_steps:
                                continue

                        if start_multisteps == end_multisteps == 1:
                            multi_step_schemes = [reaction['condition']['ReactionSmiles']]
                            if isinstance(multi_step_schemes, list):
                                if len(multi_step_schemes) == 1:
                                    if not isinstance(multi_step_schemes[0], str):
                                        continue
                                    if not (bool(re.search(r'[A-Z]', multi_step_schemes[0])) or bool(re.search(r'[a-z]', multi_step_schemes[0]))):
                                        continue
                                else:
                                    continue
                        else:
                            multi_step_schemes = reaction['condition']['Multi_step_scheme']

                        multisteps = reaction['condition']['multisteps']

                        # 反应条件或者反应式不存在，则删除该数据
                        if not multi_step_schemes or not multisteps:
                            continue
                        # 反应条件和反应步数对不上，删除该数据
                        if len(multisteps) != len(multi_step_schemes):
                            continue

                        # step += multistep
                        # 多步反应数据，但是反应条件和反应步数对不上，删除该数据
                        if number_of_reaction_steps > 1 and multi_step_schemes and multisteps:
                            multi_step_schemes_modify = select_by_multi_step_schemes(multi_step_schemes)
                            if multi_step_schemes_modify is None:
                                continue
                            reaction['condition']['Multi_step_scheme'] = multi_step_schemes_modify

                            # 处理反应条件乱标记的情况
                            multisteps_modify = modify_multisteps(multisteps)
                            if multisteps_modify is None:
                                continue
                            reaction['condition']['multisteps'] = multisteps_modify
                            # 添加参考文献id
                            reaction['reference'] = reference
                            step += 1
                            need_reactions.append(reaction)
                        elif number_of_reaction_steps == 1 and multi_step_schemes and multisteps:
                            if not multisteps[0]['Yield']:
                                continue

                            # 单步反应挑选产率最高的
                            reactionid = reaction['condition']['ReactionID']

                            multi_step_schemes_modify = select_by_multi_step_schemes(multi_step_schemes)
                            if multi_step_schemes_modify is None:
                                continue
                            reaction['condition']['ReactionSmiles'] = multi_step_schemes_modify

                            # 处理反应条件乱标记的情况
                            multisteps_modify = modify_multisteps(multisteps)
                            if multisteps_modify is None:
                                continue
                            reaction['condition']['multisteps'] = multisteps_modify
                            # 添加参考文献id
                            reaction['reference'] = reference
                            step += 1

                            try:
                                if reactionid in need_reactions_dict:
                                    old_reaction = need_reactions_dict[reactionid]
                                    old_yeild = old_reaction['condition']['multisteps'][0]['Yield'].strip('%')
                                    new_yeild = multisteps_modify[0]['Yield'].strip('%')
                                    if float(old_yeild) < float(new_yeild):
                                        need_reactions_dict[reactionid] = reaction
                                else:
                                    need_reactions_dict[reactionid] = reaction
                            except:
                                continue

    # 打印出挑选数据的条数(多步195760)
    print(step)

    output_json_path = output_path
    if start_multisteps == 1 and end_multisteps == 1:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            for key,need_jsondict in need_reactions_dict.items():
                json.dump(need_jsondict, f, ensure_ascii=False)
                f.write('\n')
    else:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            for need_jsondict in need_reactions:
                json.dump(need_jsondict, f, ensure_ascii=False)
                f.write('\n')


def generate_relation_description(multi_step_schemes, multisteps):
    reaction_conditions_descriptions = []

    # nodeset = set()
    triples_strs = []

    # 记录最终产物，用于添加关系的历史属性，便于互信息的计算
    try:
        final_products = multi_step_schemes[-1]['smiles'].split('>>')[-1].split('△')
    except:
        final_products = multi_step_schemes[-1].split('>>')[-1].split('△')

    for idx, multistep in enumerate(multisteps):
        # 得到总体底物描述和smiles式子描述
        try:
            multi_step_scheme = multi_step_schemes[idx]['smiles']
        except:
            multi_step_scheme = multi_step_schemes[idx]

        heads_str, tails_str = multi_step_scheme.split('>>')

        heads = heads_str.split('△')
        tails = tails_str.split('△')

        head_description = f'Using {" and ".join(heads)} as substrate,'
        tail_description = f'ultimately yielding product {" and ".join(tails)}, the reaction equation is {multi_step_scheme.replace("△", ".")}.'

        description = ''
        # 多步反应中的一个单步反应可能包含多个反应阶段，多个反应阶段分开来生成数据
        stages = multistep['stages']
        is_single_stage = True
        if len(stages) > 1:
            is_single_stage = False
        for stage in stages:
            if is_single_stage:
                stage_number = ''
            else:
                stage_number = 'Stage' + str(stage['stage_number']) + ':'

            description += f'{stage_number}'

            data = stage['data']
            # 1.溶剂(有多种)
            solvent = data['Solvent']
            if isinstance(solvent, str):
                solvent = [solvent]
            if solvent:
                if len(solvent) > 1:
                    description += ' and '.join(solvent) + ' as solvent, '
                else:
                    description += solvent[0] + ' as solvent, '

            # 2.试剂(有多种)
            reagent = data['Reagent']
            if reagent:
                if len(reagent) > 1:
                    description += ' and '.join(reagent) + ' as reagent, '
                else:
                    description += reagent[0] + ' as reagent, '

            # 3.催化剂(有多种)
            catalyst = data['Catalyst']
            if catalyst:
                if len(catalyst) > 1:
                    description += ' and '.join(catalyst) + ' as catalyst, '
                else:
                    description += catalyst[0] + ' as catalyst, '

            # 4.反应温度(有多种,或的关系)
            temperature = data['Temperature']
            # 5.反应时间
            time = data['Time']

            if temperature:
                if time:
                    # 时间和温度都存在
                    if len(temperature) == len(time):
                        # 时间和温度一一对应
                        for idx, single_temperature in enumerate(temperature):
                            single_time = time[idx]
                            if idx == 0:
                                description += f'The reaction is carried out at {single_temperature} for {single_time},'
                            else:
                                description += f'and then at {single_temperature} for {single_time},'
                    elif len(temperature) > 1 and len(time) == 1:
                        description += f'At {temperature[0]}, the substrate are added, then the reaction at temperatures of {" and ".join(temperature[1:])} for {time[0]}'
                    elif len(temperature) == 1 and len(time) > 1:
                        # 一个温度对应多个时间(时间加和)
                        description += f'The reaction is carried out at {temperature[0]} for {" plus ".join(time)},'
                    else:
                        print(f'时间温度同时存在的额外情况{temperature},{time}')
                else:
                    # 只有温度没有时间
                    description += f'The reaction is carried out at {temperature[0]},'
            else:
                if time:
                    # 只有时间没有温度,经测试，没有出现没有温度只有时间，且时间超过一个的情况
                    description += f'The reaction is carried out at for {time[0]},'
                else:
                    # 温度和时间都不存在
                    pass

            # 6.ph值
            ph = data['Ph']
            if ph:
                if len(ph) > 1:
                    ph = list(set(ph))
                    description += 'at ph ' + ' and '.join(ph) + ','
                else:
                    description += 'at ph ' + ph[0] + ','

            # 7.反应压力
            pressure = data['Pressure']
            if pressure:
                if len(pressure) > 1:
                    pressure = list(set(pressure))
                    description += ' at the pressure of ' + ' or '.join(pressure) + ','
                else:
                    description += ' at the pressure of ' + pressure[0] + ','

            # 8.其余条件(惰性气氛等等)todo:统计一下其余条件的种类
            otherConditions = data['OtherConditions']
            if otherConditions:
                if len(otherConditions) > 1:
                    otherConditions = list(set(otherConditions))
                    description += ' and ' + ','.join(otherConditions) + ', etc.'
                else:
                    description += ' and ' + otherConditions[0] + '.'

        # 存储一个步骤的描述
        complete_description = head_description + description + tail_description

        complete_description = complete_description.replace('  ', ' ')
        reaction_conditions_descriptions.append(complete_description)

        # 生成三元组
        for head in heads:
            for tail in tails:
                # if head == 'CCCCCCCCCCCCCCCCI' and tail=='CCCCCCCCCCCCCCCCc1ccco1':
                #     print(1)
                if tail == head:
                    print('error')
                if tail in final_products:
                    triples_str = f'{head}|PRODUCT|{tail}|{complete_description}|'
                else:
                    triples_str = f'{head}|PRODUCT|{tail}|{complete_description}|{",".join(final_products)}'
                triples_strs.append(triples_str)

    return reaction_conditions_descriptions, triples_strs


def generate_head_tail(input_multi_step_schemes: list):
    head_and_tails = []

    for smiles in input_multi_step_schemes:
        heads_str, tails_str = smiles['smiles'].split('>>')

        heads = heads_str.split('△')
        tails = tails_str.split('△')

        # 这里是smile式子标准化错误
        if None in heads or None in tails:
            return None

        # 遍历列表，生成拆分后的组合
        head_dicts = []
        if len(heads) == 1:
            head_dicts.append({'single': heads[0], 'rest': [], 'tails': tails})
        else:
            for i in range(len(heads)):
                # 当前元素作为单独的子列表
                single = heads[i]
                # 其余元素作为另一个子列表
                rest = heads[:i] + heads[i + 1:]
                # 将两个子列表组合成一个元组并添加到结果中
                head_dicts.append({'single': single, 'rest': rest, 'tails': tails})

                # 反应物也在底物中出现，予以剔除，在挑选多部反应数据的那里已经筛选了一次。这里可能是smiles标准化之后，出现的重复元素，一样不考虑催化剂的形式
                if single in tails:
                    return None

        head_and_tails.append(head_dicts)

    return head_and_tails


def generate_triples(inputpath, csv_outputpath, train_data_outputpath):
    """
    生成图谱构建所需要的csv文件
    以及训练数据生成所需要的json文件
    将图谱的条件和训练集的条件保持一致
    """
    # total_nodeset = set()
    total_triples = []
    train_data_need_dicts = []

    with open(inputpath, 'r', encoding='utf-8') as files:
        all_lines = files.readlines()
        for file in tqdm(all_lines, desc='🚀 三元组生成', unit='entry', colour='#00ff00'):
            json_file = json.loads(file)

            condition = json_file['condition']
            multisteps = condition['multisteps']

            if start_multisteps == 1 and end_multisteps == 1:
                multi_step_schemes = condition['ReactionSmiles']
            else:
                multi_step_schemes = condition['Multi_step_scheme']

            # 生成三元组条件描述
            reaction_conditions_descriptions, triples_strs = generate_relation_description(multi_step_schemes, multisteps)
            if not reaction_conditions_descriptions:
                continue
            # total_nodeset.update(nodeset)
            total_triples += triples_strs

            # 保存生成训练数据所需的信息
            json_file['condition_descriptions'] = reaction_conditions_descriptions
            train_data_need_dicts.append(json_file)

    triples = pd.DataFrame(total_triples)
    triples.to_csv(csv_outputpath, index=False)

    outputpath = csv_outputpath[:-4] + '_test.csv'
    triples_test = pd.DataFrame(triples[:1000])
    triples_test.to_csv(outputpath, index=False)

    # 这里是后续训练数据生成所需要的数据
    with open(train_data_outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in train_data_need_dicts:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def triple2csv(file_root_path,input_file_path):
    """
    生成两个csv文件，node文件：包含node名称和id，关系文件：id1 关系 id2
    """
    nodes_name2id = {}
    relation_id2name2id = {}
    relation_name2id = {}
    csv_dataset = pd.read_csv(f'{file_root_path}/{input_file_path}', sep='\t')
    # csv_dataset = [pd.read_csv(f'{file_root_path}/select_multistep_reaction_triple.csv', sep='\t').iloc[0][0]]

    for i, row in tqdm(csv_dataset.iterrows(), total=len(csv_dataset)):
        tri = row[0]
        head, relation, tail, relation_description, history = tri.split('|')

        # if head == 'CCCCCCCCCCCCCCCCI' and tail == 'CCCCCCCCCCCCCCCCc1ccco1':
        #     print(1)

        head = head.strip()
        tail = tail.strip()

        if relation_description in relation_name2id:
            edeg_attr_id = relation_name2id[relation_description]
        else:
            edeg_attr_id = len(relation_name2id)
            relation_name2id[relation_description] = edeg_attr_id

        if head not in nodes_name2id:
            nodes_name2id[head] = len(nodes_name2id)
        if tail not in nodes_name2id:
            nodes_name2id[tail] = len(nodes_name2id)
        relation_id2name2id_key = f'{nodes_name2id[head]}|{relation_description}|{nodes_name2id[tail]}'
        if relation_id2name2id_key not in relation_id2name2id:
            hset = set()
            for a in history.split(','):
                if not isinstance(a, str) or a == ' ' or a == '':
                    continue
                hset.add(a)
            relation_id2name2id[relation_id2name2id_key] = {'relation_id': edeg_attr_id, 'history': hset}
        else:
            for a in history.split(','):
                if not isinstance(a, str) or a == ' ' or a == '':
                    continue
                relation_id2name2id[relation_id2name2id_key]['history'].add(a)

    nodes_name2id = pd.DataFrame(nodes_name2id.items(), columns=['node_name', 'node_id'])
    nodes_name2id.to_csv(f'{file_root_path}/nodes.csv', index=False, columns=['node_id', 'node_name'])

    rows = []
    for outer_key, inner_dict in relation_id2name2id.items():
        history = ','.join(inner_dict['history'])
        if not isinstance(history, str) or history is None or history == '':
            history = ''
        else:
            history = f'This reaction has been used to generate {history}'
        # if history == 'This reaction has been used to generate ':
        #     print(1)
        row = {
            'relation': outer_key,
            'relation_id': inner_dict['relation_id'],
            'history': history
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{file_root_path}/relation_id2name2id.csv', index=False)
    # relation_id2name2id = pd.DataFrame(relation_id2name2id.items(), columns=['relation', 'relation_id'])
    # relation_id2name2id.to_csv(f'{file_root_path}/relation_id2name2id.csv', index=False, columns=['relation', 'relation_id'])

    print('csv文件生成完成！！！')


def csv2neo4j(node_file, relation_file):
    # 连接到 Neo4j
    graph = Graph("http://172.31.169.49:7474", auth=("neo4j", "939924"))
    meta_node = Node(
        'Meta',
        id='meta-001',
        title='myrag meta node',
        text='store some meta info',
        # python获取当前时间
        timestamp=str(datetime.now()),
        version=0,
        status='active',
    )
    graph.create(meta_node)

    nodes_dict = {}

    # 创建节点
    nodes = pd.read_csv(node_file)
    for _, node in tqdm(nodes.iterrows(), total=len(nodes)):
        name = str(node[1])
        id = str(node[0])  # 确保ID是字符串形式
        neo_node = Node('entity', name=name, id=id)
        graph.create(neo_node)
        nodes_dict[id] = neo_node  # 存储创建的节点对象以便后续关系创建使用
    print('节点创建完毕！！！！')

    with open('/code/KGLLM2.0/MyRAG/dataset/mydataset/nodesdict.json', 'w', encoding='utf-8') as f:
        json.dump(nodes_dict, f, ensure_ascii=False, indent=4)

    # with open('/code/KGLLM2.0/MyRAG/dataset/mydataset/nodesdict.json', 'r', encoding='utf-8') as f:
    #     nodes_dict = json.load(f)

    # 创建关系
    relations = pd.read_csv(relation_file)
    # relations = relations[71308:]
    for _, rel in tqdm(relations.iterrows(), total=len(relations)):
        head_id, relation, tail_id = rel[0].split('|')
        history = rel[2]
        if not isinstance(history, str):
            history = ''

        head_id = str(head_id)
        tail_id = str(tail_id)

        if head_id in nodes_dict and tail_id in nodes_dict:
            # 原始创建
            head_node = nodes_dict[head_id]
            tail_node = nodes_dict[tail_id]

            # csv创建
            # head_node_dict = nodes_dict[head_id]
            # tail_node_dict = nodes_dict[tail_id]
            #
            # # 查询是否存在同名节点
            # head_name = head_node_dict['name']
            # head_node = graph.nodes.match('entity').where(f"_.{'name'}  = '{head_name}'").first()
            #
            # tail_name = tail_node_dict['name']
            # tail_node = graph.nodes.match('entity').where(f"_.{'name'}  = '{tail_name}'").first()
            #
            # if not head_node:
            #     head_node = Node('entity', name=head_name, id=head_node_dict['id'])
            # if not tail_node:
            #     tail_node = Node('entity', name=tail_name, id=tail_node_dict['id'])

            # 创建关系
            rel_between_nodes = Relationship(head_node, 'PRODUCT', tail_node, condition=str(relation), history=history, rel_id=rel[1])
            graph.create(rel_between_nodes)
        else:
            print(f"未能找到头或尾节点: {head_id} 或 {tail_id}")

    print('关系创建完毕！！！！')


def generate_train_data(inputpath, outputpath):
    reaction_ids = []

    with open(inputpath, 'r', encoding='utf-8') as files:
        all_lines = files.readlines()

        train_data_dicts = []

        # desc='🚀 Processing Reactions'：添加进度描述和表情符号
        # unit='entry'：显示单位条目
        # colour='#00ff00'：设置进度条颜色为亮绿色
        # ncols=100：控制进度条显示宽度
        for file in tqdm(all_lines, desc='🚀 训练数据生成', unit='entry', colour='#00ff00'):
            json_file = json.loads(file)
            reference = json_file['reference']
            condition = json_file['condition']
            reaction_id = condition['ReactionID']
            # 同一个反应只需要一个
            if reaction_id in reaction_ids:
                continue
            else:
                reaction_ids.append(reaction_id)

            multisteps = condition['multisteps']
            multi_step_schemes = condition['Multi_step_scheme']
            steps = condition['Number_of_reaction_steps']
            condition_descriptions = json_file['condition_descriptions']
            assert len(multisteps) == len(multi_step_schemes) == len(condition_descriptions)

            primary_substrates = []
            required_substrates = []
            products = []
            for idx, multi_step_scheme in enumerate(multi_step_schemes):
                smiles = multi_step_scheme['smiles']
                heads_str, tails_str = smiles.split('>>')

                heads = heads_str.split('△')
                tails = tails_str.split('△')

                if idx == 0:
                    primary_substrates = heads
                required_substrates += heads

                if idx == len(multi_step_schemes) - 1:
                    assert idx > 0
                    products = tails

            # 生成数据(改写四版本)
            # 1.初始
            # I would like to obtain the reaction product G. Given the required partial reaction substrates (a, b, c), please help me supplement the remaining required
            # reaction substrates and provide the reaction pathway as well as the conditions required for the reaction
            # 我想获得反应产物G。鉴于所需的部分反应底物（a、b、c），请帮助我补充剩余的所需反应底物，并提供反应途径和反应所需的条件
            # 2.被动语态重组
            # To synthesize compound G, substrates a, b, and c have been provided as initial materials. Please determine the additional substrates needed to complete
            # the reaction sequence, along with the detailed mechanistic pathway and specific experimental conditions required for each step.
            # 为了合成化合物G，提供了底物a、b和c作为初始材料。请确定完成反应序列所需的额外底物，以及每个步骤所需的详细机理途径和具体实验条件。
            # 3.目标导向型表达
            # Given substrates a, b, and c as starting materials, propose the missing reagents required for the synthesis of product G. Include a complete reaction
            # mechanism and provide reaction conditions for each stage
            # 给定底物a、b、c作为起始材料，提出实现产物G合成所需的缺失试剂。包括完整的反应机理，并给出每个阶段的反应条件
            # 4.流程分解式结构
            # The goal is to produce the target substance G using substrates a, b, and c. Determine the required supplementary reagents for the reaction sequence
            # and provide a step-by-step approach with relevant conditions to achieve optimal synthesis.
            # 目标是使用底物a、b、c生产目标物质G。确定反应序列所需的补充试剂，并提供具有相关条件的逐步途径，以实现最佳合成。
            question1 = f'I would like to obtain the reaction product {str(products)}. Given the required partial reaction substrates {str(primary_substrates)},' \
                        f'please help me supplement the remaining required reaction substrates and provide the reaction pathway as well as the conditions required for the reaction.'
            question2 = f'To synthesize compound {str(products)}, substrates {str(primary_substrates)} have been provided as initial materials. Please determine the additional ' \
                        f'substrates needed to complete the reaction sequence, along with the detailed mechanistic pathway and specific experimental conditions required for each step.'
            question3 = f'Given substrates {str(primary_substrates)} as starting materials, propose the missing reagents required for the synthesis of product {str(products)}.' \
                        f' Include a complete reaction mechanism and provide reaction conditions for each stage.'
            question4 = f'The goal is to produce the target substance {str(products)} using substrates {str(primary_substrates)}. Determine the required supplementary reagents ' \
                        f'for the reaction sequence and provide a step-by-step approach with relevant conditions to achieve optimal synthesis.'
            questions = [question1, question2, question3, question4]

            # remaining_substrates_answer = str(required_substrates)
            answer = f'The required substrates are {" and ".join(required_substrates)}, involving a total of {len(multisteps)} reaction steps.\n'
            # answer += 'Reaction pathway:\n'
            for idx, reaction_path in enumerate(condition_descriptions):
                answer += f'Step{idx + 1}:{reaction_path}\n'

            for question in questions:
                train_data_dict = {'reactionid': reaction_id,
                                   'question': question,
                                   'answer': answer,
                                   'primary_substrates': primary_substrates,
                                   'product': products,
                                   'steps': steps,
                                   'reference': reference,
                                   'multisteps': multisteps}
                train_data_dicts.append(train_data_dict)

    # 输出
    with open(outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in train_data_dicts:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def generate_test_data(inputpath, outputpath):
    test_data_dicts = []
    seperate_by_steps_dict = {2: [], 3: [], 4: [], 5: []}

    with open(inputpath, 'r', encoding='utf-8') as files:
        for file in files:
            json_dict = json.loads(file)
            steps = json_dict['steps']
            seperate_by_steps_dict[steps].append(json_dict)

    for key, values in seperate_by_steps_dict.items():
        id_set = set()
        random_nums = []
        while True:
            random_num = random.randint(0, len(values))
            id = values[random_num]['reactionid']
            if id not in id_set:
                id_set.add(id)
                random_nums.append(random_num)
            if len(random_nums) == 500:
                break

        random_data = [values[random_num] for random_num in random_nums]
        test_data_dicts.append(random_data)

    # 输出
    with open(outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in test_data_dicts:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def modify_test_data(inputpath, outputpath):
    """
    补充iupac和分子式
    """
    need_json_dicts = []
    testid = 1
    with open(inputpath, 'r', encoding='utf-8') as files:
        for file in files:
            json_dicts = json.loads(file)
            need_dicts = []
            for json_dict in tqdm(json_dicts, desc='🚀 测试数据更改', unit='entry', colour='#00ff00'):
                smiles = eval(json_dict['answer'].split('\n')[0][28:]) + json_dict['product']

                iupacs = []
                formulas = []
                for smile in smiles:
                    iupac = smiles_to_iupac(smile)
                    if iupac['status'] == 'error':
                        iupacs.append(None)
                        formulas.append(None)
                    else:
                        iupacs.append(iupac['data'])
                        formula = iupac_to_formula(iupac['data'])
                        if formula['status'] == 'error':
                            formulas.append(None)
                        else:
                            formulas.append(formula['data'])

                json_dict['smiles'] = smiles
                json_dict['iupacs'] = iupacs
                json_dict['formulas'] = formulas
                assert len(smiles) == len(iupacs)

                json_dict['testid'] = testid
                # print(str(smiles) + '\n' + str(iupacs))
                testid += 1
                need_dicts.append(json_dict)
            need_json_dicts.append(need_dicts)

    with open(outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in need_json_dicts:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def fetch_pubchem_data(request_url):
    try:
        response = requests.get(request_url)
        response.raise_for_status()
        return {"data": response.json(), "status": "success"}
    except requests.exceptions.RequestException as e:
        return {"data": str(e), "status": "error"}


def iupac_to_formula(iupac_name):
    url = f"{base_url}/compound/name/{quote(iupac_name)}/JSON"
    data = fetch_pubchem_data(url)
    if data["status"] != "success":
        return {"data": data["data"], "status": "error"}

    try:
        properties = data["data"]["PC_Compounds"][0]["props"]
        for prop in properties:
            if prop["urn"]["label"] == "Molecular Formula":
                return {"data": prop["value"]["sval"], "status": "success"}
        return {"data": "未找到分子式", "status": "error"}
    except (KeyError, IndexError):
        return {"data": "数据解析错误: 无法提取分子式", "status": "error"}


def smiles_to_iupac(smiles):
    url = f"{base_url}/compound/smiles/{quote(smiles)}/property/IUPACName/JSON"
    data = fetch_pubchem_data(url)
    if data["status"] != "success":
        return {"data": data["data"], "status": "error"}
    try:
        return {"data": data["data"]["PropertyTable"]["Properties"][0]["IUPACName"], "status": "success"}
    except (KeyError, IndexError):
        return {"data": "未找到 IUPAC 名称", "status": "error"}


def modify_train_data():
    # train:783040
    # train_modify:782553
    # delete:487
    test_set = set()
    with open(f'{root_path}/test_data_{start_multisteps}_{end_multisteps}_modify.json', 'r', encoding='utf-8') as files:
        for file in files:
            json_dicts = json.loads(file)
            for json_dict in json_dicts:
                id0 = json_dict['reactionid']
                id1 = json_dict['reference']['id']

                id = str(id0) + '|' + str(id1)
                test_set.add(id)

    need_json_dicts = []
    with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}.json', 'r', encoding='utf-8') as files:
        num = 0
        for file in files:
            num += 1
            json_dicts = json.loads(file)
            id0 = json_dicts['reactionid']
            id1 = json_dicts['reference']['id']
            id = str(id0) + '|' + str(id1)
            if id not in test_set:
                need_json_dicts.append(json_dicts)
        print(num)
        print(len(need_json_dicts))
        print(num - len(need_json_dicts))

    with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_modify.json', 'w', encoding='utf-8') as f:
        for need_jsondict in need_json_dicts:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def select_multipathing_train_data(inputpath, outputpath):
    """
    找出多路径的数据，目前统计只有310条
    """
    from collections import defaultdict
    import re

    def group_by_keys(data_list):
        """复合键分组函数，支持key2与key3的双重校验"""
        grouped = defaultdict(list)
        for item in data_list:
            # 构建复合键：元组存储不可变键值
            composite_key = (tuple(item['primary_substrates']), tuple(item['product']))
            grouped[composite_key].append(item)
        return list(grouped.values())

    with open(inputpath, 'r', encoding='utf-8') as files:
        json_dicts = []
        for file in files:
            json_dict = json.loads(file)
            json_dicts.append(json_dict)

        multipathing_train_data_all = group_by_keys(json_dicts)
        multipathing_train_data = []
        for multipathing_train_datum in multipathing_train_data_all:
            if len(multipathing_train_datum) > 1:
                answer1 = multipathing_train_datum[0]['answer']
                answer2 = multipathing_train_datum[1]['answer']
                pattern = r"\b\d+\.([A-Za-z][^\n,]+?)\b(?=,)"
                matches1 = str(re.findall(pattern, answer1))
                matches2 = str(re.findall(pattern, answer2))

                if matches1 == matches2:
                    continue

                multipathing_train_data.append({'data': multipathing_train_datum})

    with open(outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in multipathing_train_data:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


def select_fast_data(inputpath, outputpath, step_num):
    from retrieval.graph_dao import GraphDao
    need_datas = []
    dao = GraphDao()
    with open(inputpath, 'r', encoding='utf-8') as files:
        all_lines = files.readlines()

        del_nodes = []
        for file in tqdm(all_lines, desc='🚀 训练数据count', unit='entry', colour='#00ff00'):
            is_count = True
            json_dicts = json.loads(file)
            all_substrates = json_dicts['answer'].split('\n')[0][28:].split(',')[0].split('and')

            for entity in all_substrates:
                entity = entity.strip()
                if entity in del_nodes:
                    is_count = False
                    break

                # 检索得到该实体对应的图节点
                node = dao.query_node('entity', name=entity).first()

                # 根据节点搜索得到相邻的一跳关系与尾节点
                ename_facts, eid_facts = dao.query_triples_by_node(node)
                if len(eid_facts) > step_num:
                    del_nodes.append(entity)
                    is_count = False
                    break

            if is_count:
                need_datas.append(json_dicts)

    with open(outputpath, 'w', encoding='utf-8') as f:
        for need_jsondict in need_datas:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    start_multisteps = 2
    end_multisteps = 5
    # 1.解压数据压缩包
    # unzip_files_in_directory(f'{root_path}/synthesis_reaction_data/', f'{root_path}/synthesis_reaction_data/')

    # 2.根据总数居挑选和修改所需的数据
    # select_and_modify_data(f'{root_path}/synthesis_reaction_data/', f'{root_path}/select_multistep_reaction_data_{start_multisteps}_{end_multisteps}.json',
    #                        start_multisteps=start_multisteps, end_multisteps=end_multisteps)

    # 3.生成三元组(以及输出在原始数据中添加了条件描述的数据，辅助后续的训练数据生成)
    generate_triples(f'{root_path}/select_multistep_reaction_data_{start_multisteps}_{end_multisteps}.json', f'{root_path}/select_multistep_reaction_triple_{start_multisteps}_{end_multisteps}.csv',
                     f'{root_path}/train_data_with_condition_descriptions_{start_multisteps}_{end_multisteps}.json')

    # # 4.csv文件传入neo4j
    triple2csv(root_path,f'select_multistep_reaction_triple_{start_multisteps}_{end_multisteps}.csv')
    csv2neo4j(f'{root_path}/nodes.csv', f'{root_path}/relation_id2name2id.csv')
    #
    # # # 5.生成训练数据
    # generate_train_data(f'{root_path}/train_data_with_condition_descriptions_{start_multisteps}_{end_multisteps}.json', f'{root_path}/train_data_{start_multisteps}_{end_multisteps}.json')
    # #
    # # # 6.生成测试数据
    # generate_test_data(f'{root_path}/train_data_2_5_fast_20.json', f'{root_path}/test_data_{start_multisteps}_{end_multisteps}_fast_20.json')
    # #
    # # # 7.测试数据修改
    # modify_test_data(f'{root_path}/test_data_{start_multisteps}_{end_multisteps}.json', f'{root_path}/test_data_{start_multisteps}_{end_multisteps}_modify.json')
    # #
    # # # 8.训练数据修改（删除测试数据）
    # modify_train_data()
    #
    # # 9.挑选训练更快的部分数据。
    # 10:{2: 87308, 3: 39380, 4: 15156, 5: 5720}===>147564
    # 20:{2: 125408, 3: 60448, 4: 24312, 5: 9180}===>219,348
    # step_num = 20
    # select_fast_data(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}.json', f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast_{step_num}.json',
    #                  step_num)

    # =====================================将10个子节点之内的数据从最新训练集中提取出来
    # fast_data_id = set()
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast.json', 'r', encoding='utf-8') as files1:
    #     for file in files1:
    #         jsondict = json.loads(file)
    #         reactionid = jsondict['reactionid']
    #         fast_data_id.add(reactionid)
    #
    # need_datas = []
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}.json', 'r', encoding='utf-8') as files2:
    #     for file in files2:
    #         jsondict = json.loads(file)
    #         reactionid = jsondict['reactionid']
    #         if reactionid in fast_data_id:
    #             need_datas.append(jsondict)
    # print(len(need_datas))
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast_10.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in need_datas:
    #         json.dump(need_jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    # =====================================将10个子节点之内的数据从最新训练集中提取出来

    # num = {}
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast_10.json', 'r', encoding='utf-8') as files:
    #     all_lines = files.readlines()
    #
    #     del_nodes = []
    #     for file in tqdm(all_lines, desc='🚀 训练数据count', unit='entry', colour='#00ff00'):
    #         json_dicts = json.loads(file)
    #         steps = json_dicts['steps']
    #         if steps in num:
    #             num[steps] += 1
    #         else:
    #             num[steps] = 1
    #     print(num)




