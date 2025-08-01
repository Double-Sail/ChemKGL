from dataset.expla_graphs import ExplaGraphsDataset
# from dataset.webqsp import WebQSPDataset
from dataset.mydata import MyDataset

load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'mydata': MyDataset,
    # 'webqsp': WebQSPDataset,
}
