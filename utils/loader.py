import os.path as osp

import random
import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


# for evaluating the performance using mini-batch data
def eval_deep(log, loader):
	# get the empirical batch_size for each mini-batch
	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap

# create graph batches
def split_data(data, batch):
	node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
	node_slice = torch.cat([torch.tensor([0]), node_slice])

	row, _ = data.edge_idx
	edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
	edge_slice = torch.cat([torch.tensor([0]), edge_slice])

	# Edge indices should start at zero for every graph.
	data.edge_idx -= node_slice[batch[row]].unsqueeze(0)
	data.__num_nodes__ = torch.bincount(batch).tolist()

	total_slices = {'edge_index': edge_slice}
	if data.x is not None:
		total_slices['x'] = node_slice
	if data.edge_attr is not None:
		total_slices['edge_attr'] = edge_slice
	if data.y is not None:
		if data.y.size(0) == batch.size(0):
			total_slices['y'] = node_slice
		else:
			total_slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

	return data, total_slices

def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)


def read_graph_data(folder, feature):
	edge_index = read_file(folder, 'A', torch.long).t()
	node_attrs = sp.load_npz(folder + f'new_{feature}_feature.npz')
	node_graph_id = np.load(folder + 'node_graph_id.npy')
	graph_labels = np.load(folder + 'graph_labels.npy')


	edge_attr = None
	x = torch.from_numpy(node_attrs.todense()).to(torch.float)
	node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
	y = torch.from_numpy(graph_labels).to(torch.long)
	_, y = y.unique(sorted=True, return_inverse=True)

	num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
	edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
	edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

	data = Data(x=x, edge_idx=edge_index, edge_attr=edge_attr, y=y)
	data, slices = split_data(data, node_graph_id)

	return data, slices


class ToUndirected:
	def __init__(self):
		pass

	def __call__(self, data):
		edge_attribute = None
		edge_idx = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_idx.max().item() + 1 if data.x is None else data.x.size(0)
		edge_idx, edge_attribute = coalesce(edge_idx, edge_attribute, num_nodes, num_nodes)
		data.edge_attr = edge_attribute
		data.edge_index = edge_idx
		return data

class FNNDataset(InMemoryDataset):
	def __init__(self, root, selected_feature='spacy', transform=None, pre_transform=None, pre_filter=None):
		self.dataset_name = 'politifact'
		self.feature = selected_feature
		self.root = root
		super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
		self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])

	@property
	def processed_dir(self):
		name = 'processed/'
		return osp.join(self.root, self.dataset_name, name)

	@property
	def raw_dir(self):
		name = 'raw/'
		return osp.join(self.root, self.dataset_name, name)

	@property
	def raw_file_names(self):
		names = ['node_graph_id', 'graph_labels']
		return ['{}.npy'.format(name) for name in names]

	@property
	def num_node_attributes(self):
		if self.data.x is None:
			return 0
		return self.data.x.size(1)

	@property
	def processed_file_names(self):
		if self.pre_filter is None:
			return f'{self.dataset_name[:3]}_data_{self.feature}.pt'
		else:
			return f'{self.dataset_name[:3]}_data_{self.feature}_prefiler.pt'

	def run_process(self):

		self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
		self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
		self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

		torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

	def __repr__(self):
		return '{}({})'.format(self.dataset_name, len(self))
