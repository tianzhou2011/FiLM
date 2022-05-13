
import torch.nn as nn
#from ele_utlis import InputTypes,DataTypes
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from layers.Embed import DataEmbedding_onlypos,DataEmbedding


class Model(nn.Module):
    def __init__(self,configs,hparam=None, quantile=[0.5]):
        super(Model, self).__init__()
        hparam= {
              # 'total_time_steps': 8 * 24,
              # 'num_encoder_steps': 7 * 24,
              # 'num_decoder_steps': 1 * 24,
              # 'out_size':128,
              # 'inner_size':128,
              'total_time_steps': configs.seq_len+configs.pred_len,
              'num_encoder_steps': configs.seq_len,
              'num_decoder_steps': configs.pred_len,
              'out_size':128,
              'inner_size':128,
            
            
              'num_epochs':750,
              'multiprocessing_workers': 6,
        
              'dropout_rate': 0.2, 
              'stacks':1,
              'blocks_per_stack':1,
              
              'duplicate':2,
              'outer_loop':2,
              'depth':2,
              'boosting_round':1,
              
              'minibatch_size': 215,
              'max_example':16,
              'T_max': 150,  # 100
              'max_gradient_norm': 100,
              'quantile':[0.5],
              'lr':1e-3,
          }
        self.hparams = hparam
        self.forecast_length = self.hparams['total_time_steps'] - self.hparams['num_encoder_steps']
        self.quantiles = quantile
        self.dropout_rate = self.hparams['dropout_rate']
        self.num_encoder_steps = self.hparams['num_encoder_steps']
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding_onlypos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        #self._column_definition = self.hparams['column_definition']
        #self.input_cols = [tup for tup in self._column_definition if tup[2] not in {InputTypes.ID, InputTypes.TIME}]

        '''
        ''build embeddings
        '''
#         self.known_embedding_list = nn.ModuleList([build_embeddings(self.hparams, col_tup) for col_tup in self.input_cols \
#                             if col_tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.OBSERVED_INPUT, InputTypes.TARGET}])
#         self.len_known_inputs = len(self.known_embedding_list)
        
#         self.observed_embedding_list = nn.ModuleList([build_embeddings(self.hparams, col_tup) for col_tup in self.input_cols \
#             if col_tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT,InputTypes.TARGET}])
#         self.len_observed_inputs = len(self.observed_embedding_list)

#         self.known_embedding_size = np.sum([col_tup[3] for col_tup in self.input_cols if col_tup[2] \
#                                        not in {InputTypes.ID, InputTypes.TIME, InputTypes.OBSERVED_INPUT, InputTypes.TARGET}])
    
#         self.observed_embedding_size = np.sum([col_tup[3] for col_tup in self.input_cols if col_tup[2]\
#                             not in {InputTypes.ID, InputTypes.TIME, InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT,InputTypes.TARGET}])
        
#         self.known_param = 0
#         self.observed_param = 0
#         if self.known_embedding_size>0:
#             self.weight_known = nn.Conv1d(self.known_embedding_size, 1, 1, stride=1)
#             self.know_param = 1
            
#         if self.observed_embedding_size>0:
#             self.observed_param = 1
#             self.weight_observed = nn.Conv1d(self.observed_embedding_size, 1, 1, stride=1)
        
        self.NBeats_decoder =  NBeats_boosting_trees_1(
                    stacks=self.hparams['stacks'],
                    blocks_per_stack=self.hparams['blocks_per_stack'],
                    forecast_length=self.forecast_length,
                    #backcast_length=self.know_param*self.hparams['total_time_steps']+(self.observed_param+1)*self.hparams['num_encoder_steps'],
                    backcast_length=self.hparams['num_encoder_steps'],
                    out_size=self.hparams['out_size'],
                    inner_size=self.hparams['inner_size'],
                    duplicate = self.hparams['duplicate'],
                    outer_loop =self.hparams['outer_loop'],
                    depth = self.hparams['depth'],
                    boosting_round =self.hparams['boosting_round'])
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        '''
        '' setup embeeding lists
        '''
#         print('inputs shape',inputs.shape)
#         print('known_inputs shape',known_inputs.shape)
#         print('observed_inputs shape',observed_inputs.shape)
#         if self.known_embedding_size>0:
#             self.known_embeddings_list = []
#             for i in range(self.len_known_inputs):
#                 e = self.known_embedding_list[i](known_inputs[Ellipsis, i].unsqueeze(-1))
#                 self.known_embeddings_list.append(e)
#             print('self.known_embeddings_list',len(self.known_embeddings_list))
#             print('self.known_embeddings_list[0].shape',self.known_embeddings_list[0].shape)
                
#             self.known_feature = torch.cat(self.known_embeddings_list, dim=-1).squeeze()
#             print('process self.known_feature shape',self.known_feature.shape)
#             self.known_feature = self.known_feature.permute(0,2,1)
#             print('permuted self.known_feature shape',self.known_feature.shape)
#             self.known_feature = self.weight_known(self.known_feature).squeeze()
#             print('weighted known feature shape',self.known_feature.shape)
#             #self.known_feature = torch.mean(torch.cat(self.known_embeddings_list, dim=-1),dim=-1).squeeze()
            
#             #self.known_feature,_ = self.V_known(self.known_feature)
#             #self.known_feature = self.known_feature.squeeze()
            
#         if self.observed_embedding_size>0:
#             self.observed_embeddings_list = []
#             for i in range(self.len_observed_inputs):
#                 e = self.observed_embedding_list[i](observed_inputs[Ellipsis, i].unsqueeze(-1))
#                 self.observed_embeddings_list.append(e)
#             print('self.observed_embeddings_list',len(self.observed_embeddings_list))
#             print('self.observed_embeddings_list[0].shape',self.observed_embeddings_list[0].shape)
            
            
#             self.observed_feature = torch.cat(self.observed_embeddings_list, dim=-1).squeeze()
#             self.observed_feature = self.observed_feature.permute(0,2,1)
#             self.observed_feature = self.weight_observed(self.observed_feature).squeeze()
        
#         if self.known_embedding_size>0 and self.observed_embedding_size>0:
#             backcasts = torch.cat((inputs.squeeze(),
#                                self.known_feature, 
#                                self.observed_feature)
#                               ,dim=-1)
#         if self.known_embedding_size>0 and self.observed_embedding_size==0:
#             backcasts = torch.cat((inputs.squeeze(),
#                                self.known_feature)
#                               ,dim=-1)
#         if self.known_embedding_size==0 and self.observed_embedding_size>0:
#             backcasts = torch.cat((inputs.squeeze(),
#                                self.observed_feature)
#                               ,dim=-1)
#         print('backcasts shape',backcasts.shape)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        print('enc_out shape',enc_out.shape)
        print('dec_out shape',dec_out.shape)
        backcasts = torch.cat((enc_out,dec_out),dim=1).view(enc_out.shape[0],-1)
        print('backcasts shape',backcasts.shape)
        #raise Exception('aaaa')
        backcasts, forcasts = self.NBeats_decoder(backcasts)
        print('processed backcasts shape',backcasts.shape)
        print('processed forcasts shape',forcasts.shape)
        raise Exception('aaaa')
        #outputs = torch.cat([forcasts.unsqueeze(-1) for _ in range(len(self.quantiles))],dim=-1)
        return forcasts#,outputs

class build_embeddings(nn.Module):
    def __init__(self, params, col_tup=None):
        super(build_embeddings, self).__init__()
        self.params = params
        self.col_tup = col_tup

        self._column_definition = self.params['column_definition']
        print(self.col_tup)
        # san check 成功 减1 失败减1D3....
        if self.col_tup not in self._column_definition:
            raise ValueError(f'feature {self.col_tup[0]} is not in feature list.')
        if self.col_tup[1] == DataTypes.REAL_VALUED:
            self.embedding = nn.Linear(1, self.col_tup[3])
        else:
            print(self.col_tup)
            self.embedding = nn.Embedding(num_embeddings=self.col_tup[-1], embedding_dim=self.col_tup[3])

    def forward(self, inputs):
        if self.col_tup[2] == InputTypes.STATIC_INPUT:
            inputs = Variable(inputs.long())
            return self.embedding(inputs)
        else:
            if self.col_tup[1] != DataTypes.REAL_VALUED:
                inputs = Variable(inputs.long())
            output = self.embedding(inputs)
            if self.col_tup[1] == DataTypes.REAL_VALUED:
                output = output.unsqueeze(-2)
            return output
        
        
        
class NBeats_boosting_trees_1(nn.Module):
    def __init__(self,
                forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 2,
                 outer_loop = 2,
                 boosting_round = 2):
            
        super(NBeats_boosting_trees_1, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth
        
        self.outer_loop = outer_loop
        self.boosting_round = boosting_round
        
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
        self.reuse_forecast_stacks = nn.ModuleList()
        self.reuse_backcast_stacks = nn.ModuleList()
        
        
        
#         stacks = []      
        for i in range(self.boosting_round):
            self.stacks.append(Parallel_multi_Tree_controled_NBeats(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth,
                                        outer_loop = self.outer_loop
                                        )
                              )
            self.reuse_forecast_stacks.append(self.basic_block_build())
            self.reuse_backcast_stacks.append(self.basic_block_build2())
            
        self.input_tranform()
    
#         self.stacks = nn.Sequential(*stacks)
    def input_tranform(self):
        self.layer_1 = nn.Linear(self.backcast_length, 512)
        self.layer_2 = nn.Linear(512,self.inner_size)
            
    def basic_block_build(self):
        # #print('forecast_length',self.forecast_length)
        # #print('inner_size',self.inner_size)
        # #print('backcast_length',self.backcast_length)
        stacks = []
        # for i in range(self.depth):
        #     stacks.append(nn.Linear(self.backcast_length, self.backcast_length))
        for i in range(self.depth):
            if i == 0:                stacks.append(nn.Linear(self.forecast_length,self.inner_size))
            else:
                stacks.append(nn.Linear(self.inner_size,self.inner_size))
            stacks.append(nn.ReLU())
        stacks.append(nn.Linear(self.inner_size, self.backcast_length))
        return nn.Sequential(*stacks)
            
    def basic_block_build2(self):
        #print('forecast_length',self.forecast_length)
        #print('inner_size',self.inner_size)
        #print('backcast_length',self.backcast_length)
        stacks = []
        for i in range(self.depth):
            if i==0:
                stacks.append(nn.Linear(self.backcast_length, self.forecast_length))
            else:
                stacks.append(nn.Linear(self.forecast_length, self.forecast_length))
            
        # for i in range(self.depth):
        #     if i == 0:
        #         stacks.append(nn.Linear(self.forecast_length,self.inner_size))
        #     else:
        #         stacks.append(nn.Linear(self.inner_size,self.inner_size))
        #     stacks.append(nn.ReLU())
        # stacks.append(nn.Linear(self.inner_size, self.backcast_length))
        return nn.Sequential(*stacks)
        
    def forward(self, x):

#         *pos_idxs
#         x =  self.batchnorm(x)
#         x = self.layer_2(self.layer_1(x))
       
        for i in range(self.boosting_round):
            #print('x shape',x.shape)
            x_tmp,y_tmp = self.stacks[i](x)
            #print('x process shape',x_tmp.shape)
            #print('y process shape',y_tmp.shape)
            if i == 0:
                y = self.reuse_backcast_stacks[i](y_tmp)
            else:
                y = y+self.reuse_backcast_stacks[i](y_tmp)
            x = x-x_tmp - self.reuse_forecast_stacks[i](y)
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
#         y =torch.mean(torch.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
    
        return x,y


class NBeats_boosting_trees(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4,
                 outer_loop = 2,
                 boosting_round = 2):
            
        super(NBeats_boosting_trees, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth

        self.outer_loop = outer_loop
        self.boosting_round = boosting_round
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
#         stacks = []      
        for i in range(self.boosting_round):
            self.stacks.append(Parallel_multi_Tree_controled_NBeats(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth,
                                        outer_loop = self.outer_loop
                                        )
                              )
    
#         self.stacks = nn.Sequential(*stacks)
            
  
        
    def forward(self, x):

#         *pos_idxs
#         x =  self.batchnorm(x)
        
       
        for i in range(self.boosting_round):             
            x_tmp,y_tmp = self.stacks[i](x)
            if i == 0:
                x = x-x_tmp
                y = y_tmp
            else:
                x = x-x_tmp
                y = y+y_tmp
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
#         y =torch.mean(torch.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
    
        return x,y
    

    

    
    
class Parallel_multi_Tree_controled_NBeats(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4,
                 outer_loop = 2):
            
        super(Parallel_multi_Tree_controled_NBeats, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth

        self.outer_loop = outer_loop
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
        

        for i in range(2**self.outer_loop-1):
            self.stacks.append(Parallel_Tri_controled_NBeats_new(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth
                                        )
                              )

        
        
    
               
    def forward(self, x):
        x_tmp,y_tmp_left,x_tmp_list_left,y_tmp_list_left = self.stacks[0](x)
        x_tmp_list_last = [x_tmp]
#         if self.outer_loop >1:
        y_tmp_list_last = [y_tmp_left]
#         else:
#             y_tmp_list_last = []
        for i in range(1,self.outer_loop):
            start_point = int(2**(i-1))
            end_point = int(2**i)
            for j in range(start_point,end_point):               
                x_tmp,y_tmp_left,x_tmp_list_left,y_tmp_list_left = self.stacks[(j-start_point)*2+end_point-1](x_tmp_list_last[j-1])
                x_tmp,y_tmp_right,x_tmp_list_right,y_tmp_list_right = self.stacks[(j-start_point)*2+end_point](x_tmp_list_last[j-1])
                x_tmp_list_last.append(x_tmp_list_left[:,:,0])
                x_tmp_list_last.append(x_tmp_list_left[:,:,1])
                x_tmp_list_last.append(x_tmp_list_right[:,:,0])
                x_tmp_list_last.append(x_tmp_list_right[:,:,1])
#                 if i == self.outer_loop-1:
                y_tmp_list_last.append(y_tmp_left)
                y_tmp_list_last.append(y_tmp_right)
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
        y =torch.mean(torch.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
        x =torch.mean(torch.stack(x_tmp_list_last,dim = 2),dim=2)
        return x,y



class Parallel_Tri_controled_NBeats_new(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4):
            
        super(Parallel_Tri_controled_NBeats_new, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        
        
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth

        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks =Parallel_NBeatsNet(stacks= self.stack_number ,
                                        blocks_per_stack= self.blocks_per_stack,
                                        forecast_length= self.forecast_length,
                                        backcast_length= self.backcast_length,
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        duplicate = self.duplicate,
                                        depth =self.depth)

        self.controler = Parallel_Controler(input_dim =  self.backcast_length,
                                            inner_dim = self.inner_size,
                                            controler_size = self.duplicate,
                                            depth = self.depth)
        self.create_pos_embedding()
        self.input_tranform()
        self.backcast_tranform()
        
    def input_tranform(self,low_rankness = 8,device = 'cuda'):
        self.layer_1 = nn.Linear(self.backcast_length, low_rankness)
        self.layer_2 =  nn.Linear(low_rankness,self.inner_size)
        
    def backcast_tranform(self,low_rankness = 8,device = 'cuda'):
        self.layer_3 = nn.Linear(self.inner_size, low_rankness)
        self.layer_4 =  nn.Linear(low_rankness,self.backcast_length)
        
        
    def create_pos_embedding(self,device = 'cuda'):
        self.scale_embedding = nn.Embedding( self.backcast_length, 1)
        self.locate_embedding = nn.Embedding( self.backcast_length, 1)
        self.noise_embedding = nn.Embedding( self.backcast_length, 1)
        self.pos_idx_set = torch.from_numpy(np.array([i for i in range( self.backcast_length)])).to(device)
        

    def forward(self, x):
#         print(x.shape)
#         print(self.layer_1)
#         print(self.layer_2)
#         x = self.layer_2(self.layer_1(x))
        
        pos_idxs =  self.pos_idx_set.repeat(x.shape[0],1)
#         print(pos_idxs.shape)
        scaling_embedding = self.scale_embedding(pos_idxs).squeeze(-1)
        locate_embedding = self.locate_embedding(pos_idxs).squeeze(-1)
        noise_embedding =  self.noise_embedding(pos_idxs).squeeze(-1)
#         print(x.shape,locate_embedding.shape,pos_idxs.shape)
#         err = torch.randn(size = noise_embedding.shape).to('cuda')
#         x = (x+locate_embedding)*torch.exp(scaling_embedding)#+ err*torch.exp(noise_embedding)
    
        model_weights =  self.controler(x)    
        backcast = x.repeat(1,self.duplicate).unsqueeze(-1) * (model_weights.view(x.shape[0],-1,1))
        
        backcast_tmp, forecast,backcast_mean, forecast_mean = self.stacks(backcast)
        backcast = backcast - backcast_tmp
#         print(backcast.shape,backcast_mean.mean)
#         print(self.layer_3)
#         print(self.layer_4)
#         backcast = self.layer_4(self.layer_3(backcast))
#         backcast_mean = self.layer_3(self.layer_3(backcast_mean))
        return backcast_mean,forecast_mean, backcast.view(x.shape[0],-1,self.duplicate), forecast.view(x.shape[0],-1,self.duplicate)
    


        
class Parallel_Controler(nn.Module):
    def __init__(self,
                input_dim,
                inner_dim,
                controler_size = 3,
                depth = 4,
                parallel = False):
        super(Parallel_Controler, self).__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.controler_size = controler_size
        self.depth = depth
        self.parallel = parallel
        
        
        self.control_softmax = nn.Softmax(dim = 2)
        
        self.stacks = self.single_router()
        
    def single_router(self):
        router = []#nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
#                 router.append(nn.Linear(self.input_dim, self.inner_dim)) 
                router.append(nn.Conv1d(
                                in_channels=self.input_dim * self.controler_size, 
                                out_channels=self.inner_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                             )
#             elif i == self.depth -1:
            else:
                router.append(nn.Conv1d(
                                in_channels=self.inner_dim * self.controler_size, 
                                out_channels=self.inner_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                )
            router.append(nn.ReLU())
        router.append(nn.Conv1d(
                                in_channels=self.inner_dim * self.controler_size, 
                                out_channels=self.input_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                             )
        return nn.Sequential(*router)
  
    def forward(self, x_input):
        x = x_input.repeat(1,self.controler_size).unsqueeze(-1)
#         print(x.shape,x_input.shape)
        flat = self.stacks(x)
        batch_size = x.shape[0]
        return self.control_softmax(flat.view(batch_size, -1, self.controler_size)).view(batch_size, -1,self.controler_size)
    
    
class Parallel_NBeatsNet(nn.Module):
    def __init__(self,
                forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 depth = 4,
                 duplicate = 1):
        
        super(Parallel_NBeatsNet, self).__init__()
        
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size
        
        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack
        
        self.depth = depth
        self.duplicate = duplicate
        
        
        blocks = []
        for i in range(self.stack_number):
 
            for block_id in range(self.blocks_per_stack):
                #if block_id == 0:
                block = Parallel_Block(self.inner_size, self.out_size,self.backcast_length,self.duplicate, self.depth)
                blocks.append(block)
#         blocks = nn.Sequential(*blocks)
        self.stacks = nn.Sequential(*blocks)
 

    def forward(self, backcast):  
#         backcast = backcast.repeat(1,self.duplicate).unsqueeze(-1)
#         print('asdfasdfasdfasdf',backcast.shape)
        backcast,forecast = self.stacks(backcast)
        backcast_mean = torch.mean(backcast.view(backcast.shape[0], -1, self.duplicate),dim = 2)
        forecast_mean = torch.mean(forecast.view(backcast.shape[0], -1, self.duplicate),dim = 2)
        return backcast, forecast,backcast_mean/(self.stack_number*self.blocks_per_stack), forecast_mean/(self.stack_number*self.blocks_per_stack)
        #return backcast, forecast,backcast_mean, forecast_mean
    
            
                
class Parallel_Block(nn.Module):

    def __init__(self, inner_dim, out_dim, input_dim,heads=1,depth = 4):
        super(Parallel_Block, self).__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.heads = heads
        self.depth = depth
        
        #self.basic_block = self.basic_block_build()
        self.basic_block = SpectralConv1d(in_channels=self.input_dim * self.heads, 
                             out_channels=self.inner_dim * self.heads, seq_len=self.inner_dim * self.heads,heads=heads,modes1=1064)
        
        self.theta_b_fc = SpectralConv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.input_dim * self.heads, seq_len=self.input_dim * self.heads,heads=heads,modes1=1064)
        
        self.theta_f_fc = SpectralConv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.input_dim * self.heads, seq_len=self.input_dim * self.heads,heads=heads,modes1=1064)
                
#         self.theta_b_fc =  nn.Conv1d(in_channels=self.inner_dim * self.heads, 
#                              out_channels=self.input_dim * self.heads, 
#                              kernel_size=1, 
#                              groups=self.heads)

#         self.theta_f_fc =  nn.Conv1d(in_channels=self.inner_dim * self.heads, 
#                              out_channels=self.out_dim * self.heads, 
#                              kernel_size=1, 
#                              groups=self.heads)

    def basic_block_build(self):
        stacks = []
        for i in range(self.depth):
            if i == 0:
                stacks.append(nn.Conv1d(in_channels=self.input_dim * self.heads, 
                             out_channels=self.inner_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads))
            else:
                stacks.append(nn.Conv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.inner_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads))
            stacks.append(nn.ReLU())
        return nn.Sequential(*stacks)
  
    def forward(self, x):
        if isinstance(x,tuple):
            y = x[1]
            x = x[0]
            #print('x shape',x.shape)
            x1 = self.basic_block(x)
            #print('x1 shape',x1.shape)
            #raise Exception('aaa')
            f = self.theta_f_fc(x1)
            b = self.theta_b_fc(x1)
            return x-b,y+f
        else:
            #print('x shape',x.shape)
            x1 = self.basic_block(x)
            #print('x1 shape',x1.shape)
            #raise Exception('aaa')
            f = self.theta_f_fc(x1)
            b = self.theta_b_fc(x1)
            return x-b,f


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,heads, modes1=0):
        super(SpectralConv1d, self).__init__()
        print('fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.heads=heads
        self.block=nn.Conv1d(in_channels=in_channels, 
                             out_channels=out_channels, 
                             kernel_size=1, 
                             groups=self.heads)
        print('heads',heads)
        if modes1>10000:
            modes2=modes1-10000
            self.index0 = list(range(0, min(seq_len//4, modes2//2)))
            self.index1 = list(range(len(self.index0),seq_len//2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len//4,modes2//2)]
            self.index = self.index0+self.index1
            self.index.sort()
        elif modes1 > 1000:
            modes2=modes1-1000
            self.index = list(range(0, seq_len//2))
            np.random.shuffle(self.index)
            self.index = self.index[:modes2]
        else:
            self.index = list(range(0, min(seq_len//2, modes1)))

        print('modes1={}, index={}'.format(modes1, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(1, 256, 1, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        #print(input.shape)
        #print(weights.shape)
        #raise Exception('aaa')
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q):
        # size = [B, L, H, E]
        if len(q.shape)==3:
            #print('q shape',q.shape)
            q=q.unsqueeze(-2)
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # batchsize = B
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        #print('x_ft',x_ft.shape)
        #print('weight',self.weights1.shape)

        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = x.squeeze(1).permute(0,2,1)
        #x = self.block(x)
        #print('x shape',x.shape)
        return x