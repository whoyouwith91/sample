from dataProcess import *
from k_gnn import DataLoader
from DummyIMDataset import DummyIMDataset
from DataPrepareUtils import my_pre_transform
from utils_functions import collate_fn # sPhysnet
import torch


class get_split_data():
   def __init__(self, config):
      super(get_split_data)
        
      self.train = pd.read_csv(os.path.join(config['dataPath'], config['dataset'], 'train.csv'), names=['SMILES', 'target'])
      self.valid = pd.read_csv(os.path.join(config['dataPath'], config['dataset'], 'train.csv'), names=['SMILES', 'target'])
      self.test = pd.read_csv(os.path.join(config['dataPath'], config['dataset'], 'train.csv'), names=['SMILES', 'target'])
        
class get_data_loader():
   def __init__(self, config):
      self.config = config
      self.name = config['dataset']
      self.model = config['model']
      if config['explicit_split']:
         all_index = torch.load('/scratch/projects/yzlab/group/temp_sx/qm9_processed/qm9_split.pt')
         self.train_index = all_index['train_index'].tolist()
         self.valid_index = all_index['valid_index'].tolist()
         self.test_index = all_index['test_index'].tolist()
         self.train_size, self.val_size = config['train_size'], config['val_size']
      else:
         self.train_size, self.val_size = config['train_size'], config['val_size']

      if self.name == None:
         raise ValueError('Please specify one dataset you want to work on!')
      self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_bond_features, self.num_i_2 = self.graph_loader()
      
      if self.model in ['VAE', 'TransformerUnsuper']:
         self.tokenizer = MolTokenizer(vocab_file=os.path.join(self.config['vocab_path'], self.config['vocab_name']))
    
         pad_idx = self.tokenizer.pad_token_id
         unk_idx = self.tokenizer.unk_token_id
         eos_idx = self.tokenizer.eos_token_id
         init_idx = self.tokenizer.bos_token_id   
           
         self.vocab_size = self.tokenizer.vocab_size
         TEXT = torchtext.data.Field(use_vocab=False, \
                                    tokenize=self.tokenizer.encode, \
                                    pad_token=pad_idx, \
                                    unk_token=unk_idx, \
                                    eos_token=eos_idx, \
                                    init_token=init_idx, \
                                    batch_first=True)
         IDS = torchtext.data.Field(use_vocab=False, dtype=torch.long, sequential=False)
         self.fields = [("id", IDS), ("SRC", TEXT), ("TRG", TEXT)]

         self.train_loader, self.val_loader, self.test_loader = self.vaeLoader()
    
   def graph_loader(self):

      if self.config['model'] in ['1-2-GNN']:
         if self.config['taskType'] != 'multi': 
            dataset = knnGraph(root=self.config['data_path'], pre_transform=MyPreTransform(), pre_filter=MyFilter())
         else:
            dataset = knnGraph_multi(root=self.config['data_path'], pre_transform=MyPreTransform(), pre_filter=MyFilter())
         dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
         num_i_2 = dataset.data.iso_type_2.max().item() + 1
         dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

      elif self.config['model'] in ['1-2-efgs-GNN', '1-efgs-GNN']:
         if self.config['taskType'] != 'multi':
            dataset = knnGraph_EFGS(root=self.config['data_path'], pre_transform=MyPreTransform_EFGS(), pre_filter=MyFilter())
         else:
            dataset = knnGraph_EFGS_multi(root=self.config['data_path'], pre_transform=MyPreTransform_EFGS(), pre_filter=MyFilter())
         dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
         num_i_2 = dataset.data.iso_type_2.max().item() + 1
         dataset.data.iso_type_2 = F.one_hot(dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)
        
      elif self.config['model'] in ['1-interaction-GNN']:
         dataset = knnGraph_WithWater(root=self.config['data_path'])
         num_features = dataset[0]['x'].shape[1]
         num_bond_features = dataset[0]['edge_attr'].shape[1]
         my_split_ratio = [self.train_size, self.val_size]  #my dataseet
         test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
         rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
         train_dataset = rest_dataset[:my_split_ratio[0]]
         val_dataset = rest_dataset[my_split_ratio[0]:]

         test_loader = DataLoader_WithWater(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
         val_loader = DataLoader_WithWater(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
         train_loader = DataLoader_WithWater(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)

         return train_loader,  val_loader, test_loader, num_features, num_bond_features, None

      elif self.config['model'] in ['physnet']:
         #if self.config['dataset'] == 'qm9':
         dataset = physnet(root=self.config['data_path'], pre_transform=my_pre_transform)
         #dataset = DummyIMDataset(root=self.config['data_path'], dataset_name='processed.pt')
         num_i_2 = None
      else: # 1-GNN
         if self.config['dataset'] in ['solNMR', 'solALogP', 'sol_calc/ALL/smaller', 'sol_calc/ALL/smaller_18W', 'sol_calc/ALL/smaller_28W', 'sol_calc/ALL/smaller_38W', 'sol_calc/ALL/smaller_48W', 'sol_calc/ALL/smaller_58W']:
            if self.config['propertyLevel'] == 'multiMol': # for multiple mol properties
               dataset = knnGraph_multi(root=self.config['data_path'])
            if self.config['propertyLevel'] == 'molecule': # naive, only with solvation property
               if self.config['gnn_type'] == 'dmpnn':
                  dataset = knnGraph_dmpnn(root=self.config['data_path'])
               else:
                  dataset = knnGraph_single(root=self.config['data_path'])
            if self.config['propertyLevel'] in ['atom', 'atomMol']: #  either for atom property only or atom/mol property 
               dataset = knnGraph_atom(root=self.config['data_path'])
            num_i_2 = None
         elif self.config['dataset'] in ['logp_calc/ALL/smaller_58W']:
            if self.config['propertyLevel'] == 'multiMol': # for multiple mol properties
               dataset = knnGraph_multi(root=self.config['data_path'])
            if self.config['propertyLevel'] == 'molecule': # naive, only with solvation property
               if self.config['gnn_type'] == 'dmpnn':
                  dataset = knnGraph_dmpnn(root=self.config['data_path'])
               else:
                  dataset = knnGraph_logp(root=self.config['data_path'])
            if self.config['propertyLevel'] in ['atom', 'atomMol']: #  either for atom property only or atom/mol property 
               dataset = knnGraph_atom(root=self.config['data_path'])
            num_i_2 = None
         elif self.config['dataset'] in ['qm9/nmr/carbon', 'qm9/nmr/carbon/smaller', 'qm9/nmr/hydrogen', 'nmr/hydrogen', 'nmr/carbon']:
            dataset = knnGraph_nmr(root=self.config['data_path'])
            num_i_2 = None
         elif self.config['mol_features']:
            pass
         elif self.config['dataset'] == 'solEFGs':
            dataset = knnGraph_solEFGs(root=self.config['data_path'])
            num_i_2 = None
         else: # for typical other datasets
            if self.config['gnn_type'] == 'dmpnn':
               dataset = knnGraph_dmpnn_exp(root=self.config['data_path'])
               num_i_2 = None
            else:
               dataset = knnGraph(root=self.config['data_path'])
               num_i_2 = None
      
      num_features = dataset.num_features
      if self.config['model'] not in ['physnet']:
         num_bond_features = dataset[0]['edge_attr'].shape[1]
      else:
         num_bond_features = 0
          
      my_split_ratio = [self.train_size, self.val_size]  #my dataseet
      if self.config['explicit_split']:
         #mean, std = dataset[self.train_index].data.mol_y.mean(), dataset[self.train_index].data.mol_y.std()
         #dataset.data.mol_y = (dataset.data.mol_y - mean) / std
         test_dataset = dataset[self.test_index]
         val_dataset = dataset[self.valid_index]
         train_dataset = dataset[self.train_index]
      else:
         test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
         rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
         train_dataset, val_dataset = rest_dataset[:my_split_ratio[0]], rest_dataset[my_split_ratio[0]:]

      if self.config['model'] in ['physnet']:
         if self.config['explicit_split']: # when using fixed data split 
            test_loader = torch.utils.data.DataLoader(dataset[self.test_index], batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(dataset[self.valid_index], batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            train_loader = torch.utils.data.DataLoader(dataset[self.train_index], batch_size=self.config['batch_size'], num_workers=0, shuffle=True, collate_fn=collate_fn)
         else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0, collate_fn=collate_fn)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True, collate_fn=collate_fn)
      else:
         if self.config['gnn_type'] == 'dmpnn':
            test_loader = DataLoader_dmpnn(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader_dmpnn(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader_dmpnn(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
         else:
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
      
      return train_loader, val_loader, test_loader, num_features, num_bond_features, num_i_2