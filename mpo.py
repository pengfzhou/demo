import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import numpy as np
import torch.nn.functional as F
import math

from utils import AttrProxy


class MPO(nn.Module):
     def __init__(self,in_feat,out_feat,array_in,array_out,bond_dim,bias=True)  :
        super(MPO,self).__init__()
        
        self.array_in=array_in
        self.array_out=array_out
        self.bond_dim=bond_dim
        self.in_feat=in_feat
        self.out_feat=out_feat
        self.define_parameters()
        if bias:
           self.bias=Parameter(torch.Tensor(out_feat))
        else:
           self.register_parameter('bias', None)
        self.reset_parameters()
     def define_parameters(self):
        self.weight=torch.nn.ParameterList([])
        for i in range(len(self.array_in)):
            if i==0:
               self.weight.append(Parameter(torch.Tensor(self.array_in[0],self.array_out[0],self.bond_dim)))
            elif i==len(self.array_in)-1:
               self.weight.append(Parameter(torch.Tensor(self.array_in[i],self.bond_dim,self.array_out[i])))
            else:
               self.weight.append(Parameter(torch.Tensor(self.array_in[i],self.bond_dim,self.array_out[i],self.bond_dim)))
     def reset_parameters(self):
        if self.bias is not None:
            fan_out=self.out_feat
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)    
        gain=1.0
        std = gain * math.sqrt(2.0 / float(self.in_feat + self.out_feat))
        a = math.sqrt(3.0) * std             
        a=0.01
        for i in self.weight:
            #print(i.shape)
            a=math.sqrt(a*math.sqrt(3.0/self.bond_dim))
            torch.nn.init.uniform_(i,-a,a)
     def forward(self,input):
        shape=self.array_in.copy()
        size=np.prod(input.shape)
        dim=int(size/input.shape[-1])
        shape.insert(0,dim)
        output=input.reshape(shape)
        shape_last=input.shape
        list_shape=[shape_last[i] for i in range(len(shape_last)-1)]
        list_shape.append(self.out_feat)
        for i in range(len(self.weight)):
            if i==0:
               output = torch.einsum('abc,bmn->amcn',output.reshape(dim,shape[1],-1),self.weight[0])
            elif i==len(self.weight)-1:
               output = torch.einsum('abcd,cbm->adm',output.reshape(dim,self.bond_dim,shape[i+1],-1),self.weight[i]).reshape(list_shape)
            else:
               output=torch.einsum('abcd,cbmn->andm',output.reshape(dim,self.bond_dim,shape[i+1],-1),self.weight[i])               
        if self.bias is not None:
            output+=self.bias
        # to be auto-contraction
        return output

class MPO2(nn.Module):
    def __init__(self, Din, Dout, bias=False, chi=2, seed=-1):
        """
        Din (and Dout) should be a tuple containing all input (output) dimensions
        """
        super(MPO2, self).__init__()
        self.Din = Din
        self.Dout = Dout
        self.bondim = [chi for i in Din]
        self.bondim[-1] = 1
        #print("Din=", Din, "Dout=", Dout)
        assert (len(self.Din) == len(self.Dout))
        self.tensors = []
        self.npin = np.prod(self.Din)
        self.npout = np.prod(self.Dout)
        gain = 1.0
        std = gain * math.sqrt(2.0 / float(self.npin+self.npout))
        std = math.sqrt(3.0) * std
        std=0.02
        if seed > 0:
            torch.manual_seed(seed)
        for i, din in enumerate(self.Din):
            std=math.sqrt(std*math.sqrt(3.0/self.bondim[i]))
            dout = self.Dout[i]
            a = torch.rand(self.bondim[i - 1], self.bondim[i], din, dout) / math.sqrt(self.npout)
            torch.nn.init.uniform_(a,-std,std)
            exec("self.tensors_" + str(i) + "=Parameter(a.clone())")
        if bias:
            self.bias = Parameter(torch.zeros([self.npout, 1]))
        else:
            self.register_parameter('bias', None)
        self.tensors = AttrProxy(self, 'tensors_')

        #print(self)
        #print("Parameters in the class")
        params = list(self.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        #print('Total number of trainable parameters: {}'.format(nparams))
        #for param in self.parameters():
            #print(type(param.data), param.size())
        

    
    def forward(self, input):
        #print(input.shape)  # (len*batch, hidden=1024)
        shape0=input.shape[0]
        shape1=input.shape[1]
        input = input.reshape(input.shape[0]*input.shape[1],1,1,self.Din[0],-1)
        for i in range(len(self.Din)):
            input = torch.einsum("bijkl,jakm->bimal",input,self.tensors[i])
            Dnext = self.Din[i+1] if i<len(self.Din)-1 else 1
            newshape=[input.shape[0],input.shape[1]*input.shape[2],input.shape[3],Dnext,-1]
            input = input.contiguous().view(newshape)
        if self.bias is not None:    
           return input.contiguous().view(shape0,shape1,-1)+self.bias
        else:
           return input.contiguous().view(shape0,shape1,-1)


class MPO3(nn.Module):
    """多层网络"""
    def __init__(self, Din, Dout, bias=False, chi=4, chi2=4, mpo3layer=2):
        super().__init__()
        self.Din = Din
        self.Dout = Dout
        self.bondim = [chi for i in Din]    # 垂直bond
        self.bondim[-1] = 1
        self.dim = chi2       # 水平bond
        print('self.bondim:', self.bondim, ',self.dim:', self.dim)
        print("Din=", Din, "Dout=", Dout)
        assert (len(self.Din) == len(self.Dout))
        self.tensors = []
        self.npin = np.prod(self.Din)
        self.npout = np.prod(self.Dout)
        self.num_layers = mpo3layer
        for j in range(self.num_layers):
            for i in range(len(Din)):
                # if j == 0:
                exec('self.tensors_' + str(i + j * len(Din)) +
                    '=nn.Parameter(torch.randn(self.bondim[i - 1], self.bondim[i],'
                    ' self.Din[i] if j == 0 else self.dim, self.Dout[i] if j == self.num_layers-1 else self.dim) / math.sqrt(self.npout))')
                # elif j == self.num_layers-1:
                #     exec('self.tensors_' + str(i + len(Din)) +
                #          '=nn.Parameter(torch.randn(self.bondim[i - 1], self.bondim[i], self.dim, self.Dout[i]) / math.sqrt(self.npout))')
                # else:
                #     exec('self.tensors_' + str(i + j * len(Din)) +
                #          '=nn.Parameter(torch.randn(self.bondim[i - 1], self.bondim[i], self.dim, self.dim) / math.sqrt(self.npout))')
        # for i in range(len(Din)):
        #     exec('self.tensors_' + str(i + len(Din)) +
        #          '=nn.Parameter(torch.randn(self.bondim[i - 1], self.bondim[i], self.dim, self.Dout[i]) / math.sqrt(self.npout))')
        if bias:
            self.bias = Parameter(torch.zeros([self.npout, 1]))
        else:
            self.register_parameter('bias', None)
        self.tensors = AttrProxy(self, 'tensors_')

        print(self)
        print("Parameters in the class")
        params = list(self.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        print('Total number of trainable parameters: {}'.format(nparams))
        for param in self.parameters():
            print(type(param.data), param.size())
        self.reset_parameters()

    def reset_parameters(self):
        tensors = [[] for i in range(self.num_layers)]
        # print(tensors)
        for j in range(self.num_layers):
            self.in_feat = args.hidden_dim if j == 0 else args.chi2**len(self.Din)
            self.out_feat = args.vocab_size if j == self.num_layers-1 else args.chi2**len(self.Din)
            self.bond_dim = args.chi
            if self.bias is not None:
                fan_out = self.out_feat
                bound = 1 / math.sqrt(fan_out)
                torch.nn.init.uniform_(self.bias, -bound, bound)
            gain = 1.0
            std = gain * math.sqrt(2.0 / float(self.in_feat + self.out_feat))
            a = math.sqrt(3.0) * std
            for i in range(len(self.Din)):
                tensors[j].append(self.tensors[i+j*len(self.Din)])
            # print(tensors)
            for i in tensors[j]:
                # print('i', i.shape)
                a = math.sqrt(a * math.sqrt(3.0 / self.bond_dim))
                torch.nn.init.uniform_(i, -a, a)


    def forward(self, input):
        input = input.reshape(input.shape[0], 1, 1, self.Din[0], -1)
        for j in range(self.num_layers):
            for i in range(len(self.Din)):
                input = torch.einsum("bijkl,jakm->bimal", input, self.tensors[i+j*len(self.Din)])
                if j == 0:
                    Dnext = self.Din[i + 1] if i < len(self.Din) - 1 else 1
                else:
                    Dnext = self.dim if i < len(self.Din) - 1 else 1
                newshape = [input.shape[0], input.shape[1] * input.shape[2], input.shape[3], Dnext, -1]
                # print('i:', i, ',input.shape:', input.size())
                input = input.contiguous().view(newshape)
            input = input.reshape(input.shape[0], 1, 1, self.dim, -1)

        # # print('input.shape:', input.size())
        # for i in range(len(self.Din)):
        #     input = torch.einsum("bijkl,jakm->bimal", input, self.tensors[i])
        #     Dnext = self.Din[i + 1] if i < len(self.Din) - 1 else 1
        #     newshape = [input.shape[0], input.shape[1] * input.shape[2], input.shape[3], Dnext, -1]
        #     # print('i:', i, ',input.shape:', input.size())
        #     input = input.contiguous().view(newshape)
        #     # print('i:', i,',newshape:', input.size())
        # input = input.reshape(input.shape[0], 1, 1, self.dim, -1)
        # # print('input.shape:', input.size())
        # for i in range(len(self.Din)):
        #     input = torch.einsum("bijkl,jakm->bimal", input, self.tensors[i + len(self.Din)])
        #     Dnext = self.dim if i < len(self.Din) - 1 else 1
        #     newshape = [input.shape[0], input.shape[1] * input.shape[2], input.shape[3], Dnext, -1]
        #     # print('i:', i, ',input.shape:', input.size())
        #     input = input.contiguous().view(newshape)
        #     # print('i:', i, ',newshape:', input.size())


        return input.contiguous().view(input.shape[0], -1)

class Operator(nn.Module):
    """
    Use sequence of operators to span a large space from the input space with a smaller dimension
    """
    __constants__ = ['bias']
    def __init__(self, Din, Dout, bias=False):
        """
        Din (and Dout) should be a tuple containing all input (output) dimensions
        """
        super().__init__()
        self.Din=Din
        self.Dout=Dout
        print("Din=",Din,"Dout=",Dout)
        assert(len(self.Din) == len(self.Dout))
        self.tensors=[]
        self.npin = np.prod(self.Din)
        self.npout = np.prod(self.Dout)
        for i in range(len(self.Din)-1):
            d0 = self.Din[i] if i==0 else self.Dout[i]
            d1 = self.Din[i+1]
            d2 = self.Dout[i]
            d3 = self.Dout[i+1]
            a=torch.rand(d0,d1,d2,d3)/math.sqrt(self.npout)
            exec("self.operators_"+str(i)+"=Parameter(a.clone())")
        self.operators = AttrProxy(self, 'operators_')
        params = list(self.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        print('Operator layer: total number of trainable parameters: {}'.format(nparams))
        for param in self.parameters():
            print(type(param.data), param.size())
        self.reset_parameters()

    def reset_parameters(self):
        self.in_feat = args.hidden_dim
        self.out_feat = args.vocab_size
        self.bond_dim = args.chi
        # if self.bias is not None:
        #     fan_out = self.out_feat
        #     bound = 1 / math.sqrt(fan_out)
        #     torch.nn.init.uniform_(self.bias, -bound, bound)
        gain = 1.0
        std = gain * math.sqrt(2.0 / float(self.in_feat + self.out_feat))
        a = math.sqrt(3.0) * std
        tensors = []
        for j in range(len(self.Din)-1):
            tensors.append(self.operators[j])
        for i in tensors:
            # print('i', i.shape)
            a = math.sqrt(a * math.sqrt(3.0 / self.bond_dim))
            torch.nn.init.uniform_(i, -a, a)

    def forward(self, input):
        input = input.reshape(input.shape[0],1,self.Din[0],self.Din[1],-1)
        for i in range(len(self.Din)-1):
            input = torch.einsum("bxijk,ijlm->bxlmk",input,self.operators[i])
            Dnext = 1 if i==len(self.Din)-2 else self.operators[i+1].shape[1]
            newshape=[input.shape[0],input.shape[1]*input.shape[2],input.shape[3],Dnext,-1]
            input = input.contiguous().view(newshape)
        return input.contiguous().view(input.shape[0],-1)


class Operator2(nn.Module):
    def __init__(self, Din, Dout, bias=False, chi=4):
        super().__init__()
        self.Din = Din
        self.Dout = Dout
        self.bondim = chi
        print('self.bondim:', self.bondim)
        print("Din=", Din, "Dout=", Dout)
        assert (len(self.Din) == len(self.Dout))
        self.tensors = []
        self.npin = np.prod(self.Din)
        self.npout = np.prod(self.Dout)
        for i in range(len(Din)-1):
            d0 = self.Din[i] if i == 0 else self.Din[i+1]
            d1 = self.Din[1] if i == 0 else self.bondim
            if i == 1:
                d2 = self.Dout[2]
            elif i == 4:
                d2 = self.Dout[5]
            else:
                d2 = self.bondim
            d3 = self.bondim if i == 2 else self.Dout[i]
            exec('self.tensors_'+str(i+1)+'=nn.Parameter(torch.randn(d0,d1,d2,d3)/math.sqrt(self.npout))')
        if bias:
            self.bias = Parameter(torch.zeros([self.npout, 1]))
        else:
            self.register_parameter('bias', None)
        self.tensors = AttrProxy(self, 'tensors_')

        print(self)
        print("Parameters in the class")
        params = list(self.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        print('Total number of trainable parameters: {}'.format(nparams))
        for param in self.parameters():
            print(type(param.data), param.size())


    def forward(self, input):
        input = input.reshape(input.shape[0],1,self.Din[0],self.Din[1],-1)
        self.order = [1,4,3,5,2]
        #
        # input = torch.einsum("bxijk,ijlm->bxlmk",input,self.tensor[1])
        # newshape = [input.shape[0], input.shape[1] * input.shape[3], self.Din[4], input.shape[2], -1]
        # input = input.contiguous().view(newshape)
        #
        # input = torch.einsum("bxijk,ijlm->bxlmk", input, self.tensor[4])
        # newshape = [input.shape[0], input.shape[1] * input.shape[3], self.Din[3], input.shape[2], -1]
        # input = input.contiguous().view(newshape)
        #
        # input = torch.einsum("bxijk,ijlm->bxlmk", input, self.tensor[3])
        # newshape = [input.shape[0], input.shape[1], self.Din[5], input.shape[2], -1]
        # input = input.contiguous().view(newshape)
        #
        # input = torch.einsum("bxijk,ijlm->bxlmk", input, self.tensor[5])
        # newshape = [input.shape[0], input.shape[1] * input.shape[2] * input.shape[3], self.Din[2], self.bondim, -1]
        # input = input.contiguous().view(newshape)
        #
        # input = torch.einsum("bxijk,ijlm->bxlmk", input, self.tensor[2])
        #
        # print('input.shape=', input.size())
        for i,j in enumerate(self.order):
            input = torch.einsum("bxijk,ijlm->bxlmk", input, self.tensors[j])
            # print('i, j=', i, j, ',input.shape=', input.size())
            if i == 4:
                break
            if i == 2:
                newshape = [input.shape[0], input.shape[1], self.Din[self.order[i+1]], self.bondim, -1]
            elif i == 3:
                newshape = [input.shape[0], input.shape[1] * input.shape[2] * input.shape[3], self.Din[self.order[i+1]], self.bondim, -1]
            else:
                newshape = [input.shape[0], input.shape[1] * input.shape[3], self.Din[self.order[i+1]], self.bondim, -1]
            input = input.contiguous().view(newshape)
            # print('i, j=', i, j, ',input.newshape=', input.size())
        return input.contiguous().view(input.shape[0], -1)