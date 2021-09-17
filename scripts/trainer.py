import sys
from helper import *
from data import *
from models import *
from utils_functions import floating_type

def train_model(config):
    if config['model'] in ['physnet']:
        return train_physnet
    else:
        return train

def test_model(config):
    if config['model'] in ['physnet']:
        return test_physnet
    else:
        return test

def cv_train(config, table):
    results = []

    for seed in [1, 13, 31]:
        #config['data_path'] = os.path.join(config['cv_path'], 'cv_'+str(i)+'/')
        #print(config)
        #print(config['data_path'])   
        set_seed(seed)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config['device'] = device

        loader = get_data_loader(config)
        train_loader, val_loader, _, std, num_features, num_bond_features, num_i_2 = loader.train_loader, loader.val_loader, loader.test_loader, loader.std, loader.num_features, loader.num_bond_features, loader.num_i_2
        config['num_features'], config['num_bond_features'], config['num_i_2'], config['std'] = int(num_features), num_bond_features, num_i_2, std 

        model = get_model(config)
        args = objectview(config)
        model_ = model.to(device)
        #num_params = param_count(model_)
        
        optimizer = get_optimizer(config['optimizer'], model_)
        if this_dic['lr_style'] == 'decay':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=0.00001)
        if this_dic['uncertainty']:
            optimizer = torchcontrib.optim.SWA(optimizer)

        for _ in range(1, config['epochs']):
            val_res, tst_res = []
            loss = train(model_, optimizer, train_loader, config)
            val_error = test(model_, val_loader, config)
            tst_error = test(model_, test_loader, config)
            val_res.append(val_error)
            tst_res.append(tst_error)
        
        results.append(tst_res[val_res.index(min(val_res))])
    
    table.add_row([config['emb_dim'], config['num_layer'], config['NumOutLayers'], config['lr'], config['batch_size'], np.mean(results), np.std(results)])
    print(table)
    sys.stdout.flush()
    print('\n')
    sys.stdout.flush()
    return np.average(results)

def train(model, optimizer, dataloader, config, scheduler=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    if config['propertyLevel'] == 'atom':
        all_atoms = 0
    if config['dataset'] == 'solEFGs':
        preds, labels = [], []
    for data in dataloader:
        data = data.to(config['device'])
        #print(data.x.shape)
        #sys.stdout.flush()
        optimizer.zero_grad()
        y = model(data) # y contains different outputs depending on the # of tasks
        
        if config['dataset'] in ['solNMR', 'solALogP', 'qm9/nmr/allAtoms', 'sol_calc/ALL/smaller', 'sol_calc/ALL/smaller_18W', 'sol_calc/ALL/smaller_28W', 'sol_calc/ALL/smaller_38W', 'sol_calc/ALL/smaller_48W', 'sol_calc/ALL/smaller_58W', 'logp_calc/ALL/smaller_58W']:
            if config['propertyLevel'] == 'molecule': # single task on regression
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.mol_sol_wat)
                if config['gnn_type'] == 'dmpnn':
                    all_loss += loss.item() * config['batch_size']
                else:
                    all_loss += loss.item() * data.num_graphs

            elif config['propertyLevel'] == 'atom': # 
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(data.atom_y, y[0])
                all_loss += loss.item() * data.N.sum()
                all_atoms += data.N.sum()
            elif config['propertyLevel'] == 'multiMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.mol_gas) + \
                       get_loss_fn(config['loss'])(y[1], data.mol_wat) + \
                       get_loss_fn(config['loss'])(y[2], data.mol_oct) + \
                       get_loss_fn(config['loss'])(y[3], data.mol_sol_wat) + \
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_oct)
                       #get_loss_fn(config['loss'])(y[5], data.mol_logp)
            elif config['propertyLevel'] == 'atomMultiMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + \
                       get_loss_fn(config['loss'])(y[1], data.mol_gas) + \
                       get_loss_fn(config['loss'])(y[2], data.mol_wat) + \
                       get_loss_fn(config['loss'])(y[3], data.mol_oct) + \
                       get_loss_fn(config['loss'])(y[4], data.mol_sol_wat) + \
                       get_loss_fn(config['loss'])(y[5], data.mol_sol_oct) + \
                       get_loss_fn(config['loss'])(y[6], data.mol_logp)
            elif config['propertyLevel'] == 'atomMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y) + get_loss_fn(config['loss'])(y[1], data.mol_sol_wat)
            else:
                 raise "LossError"
        
        elif config['dataset'] == 'solEFGs': # solvation for regression and EFGs labels for classification 
            if config['propertyLevel'] == 'atomMol':
                assert config['taskType'] == 'multi'
                loss = get_loss_fn(config['atom_loss'])(y[0], data.atom_y) + get_loss_fn(config['mol_loss'])(y[1], data.mol_sol_wat)
            elif config['propertyLevel'] == 'atom':
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[0], data.atom_y)
                idx = F.log_softmax(model(data)[0], 1).argmax(dim=1)
                preds.append(idx.detach().data.cpu().numpy())
                labels.append(data['atom_y'].detach().data.cpu().numpy())
            else:
                raise "LossError"
        
        else: 
            if config['propertyLevel'] == 'molecule': # for single task, like exp solvation, solubility, ect
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(y[1], data.mol_y)
                if config['gnn_type'] == 'dmpnn':
                    all_loss += loss.item() * config['batch_size']
                else:
                    all_loss += loss.item() * data.num_graphs
            elif config['propertyLevel'] == 'atom': # Exp nmr/carbon, nmr/hydrogen
                assert config['taskType'] == 'single'
                loss = get_loss_fn(config['loss'])(data.atom_y, y[0], data.mask)
                all_atoms += data.mask.sum()
                all_loss += loss.item() * data.mask.sum()
            else:
                raise "LossError"
 
        loss.backward()
        if config['clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if config['scheduler'] == 'NoamLR':
            scheduler.step()
    if config['optimizer'] in ['SWA']:
        optimizer.swap_swa_sgd()
    
    if config['propertyLevel'] == 'atom':
        if config['dataset'] == 'solEFGs':
            return loss.item(), get_metrics_fn(config['metrics'])(np.hstack(labels), np.hstack(preds))
        else:
            return all_loss.item() / all_atoms.item() # MAE
    if config['propertyLevel'] == 'molecule':
        if config['metrics'] == 'l2':
            return np.sqrt(all_loss / len(dataloader.dataset)) # RMSE
        else:
            return all_loss / len(dataloader.dataset) # MAE
    if config['propertyLevel'] in ['atomMol', 'multiMol', 'atomMultiMol']:
        return loss.item()