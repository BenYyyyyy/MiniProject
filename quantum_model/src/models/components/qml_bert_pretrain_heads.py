from transformers import BertModel, BertTokenizer
from transformers import BertConfig
import torch 
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import pennylane as qml
import math
import sys
from torch import nn
from torch.nn import CrossEntropyLoss
import math
import itertools

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainingHeads4QML(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads4QML, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        
        self.num_qubits = config.n_qubits
        self.num_entangle_circuit_layers = config.n_entangling_circuit_layers
        self.num_unitary_circuit_layers = config.n_unitary_circuit_layers

        #angle_embedding_layer_shape = 2
        #entangle_params_shape = (3, 2) 
        #self.angle_embedding_params_transform_layer = torch.nn.Linear(config.hidden_size, self.num_qubits)
        self.entangle_params_transform_layer = torch.nn.Linear(config.hidden_size, self.num_entangle_circuit_layers * self.num_qubits)
        self.unitary_params = torch.nn.Parameter(torch.Tensor(self.num_unitary_circuit_layers, self.num_qubits, 3))
        # self.unitary_params = torch.nn.Parameter(torch.Tensor(self.num_unitary_circuit_layers, self.num_qubits))
        self.out =  nn.Linear(2**self.num_qubits, 2)

        # self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.quantum_layer = self.get_circuit()

    def get_circuit(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)
        @qml.qnode(dev, interface='torch')
        def quantum_model(w, unitary_params):
            #qml.templates.AngleEmbedding(x, wires=range(self.num_qubits))
            qml.templates.BasicEntanglerLayers(w, wires=range(self.num_qubits))
            #qml.templates.StronglyEntanglingLayers(w, wires=range(self.num_qubits))
            qml.templates.StronglyEntanglingLayers(unitary_params, wires=range(self.num_qubits))
            # qml.templates.BasicEntanglerLayers(unitary_params, wires=range(self.num_qubits))
            return qml.probs(wires=range(self.num_qubits))
        return quantum_model
    
    def forward(self, sequence_output):
        # bs,num_tokens,_ = sequence_output.shape
        # prediction_scores = self.predictions(sequence_output.view(bs*num_tokens,-1)).view(bs,num_tokens,-1)
        
        prediction_scores = self.predictions(sequence_output)

        # seq_relationship_score = self.seq_relationship(pooled_output)
        
        #angle_embedding_params = self.angle_embedding_params_transform_layer(pooled_output)
        #angle_embedding_params = torch.sigmoid(angle_embedding_params)* 2 * math.pi
        entangle_params = self.entangle_params_transform_layer(sequence_output[:, 0])
        entangle_params = torch.sigmoid(entangle_params.reshape(-1, self.num_unitary_circuit_layers, self.num_qubits))* 2 * math.pi
        
        
        qcircuit_output = self.quantum_layer(entangle_params, self.unitary_params)
        #qcircuit_output = self.quantum_layer(entangle_params)
        
        seq_relationship_score = self.out(qcircuit_output.float())
        # seq_relationship_score = torch.tensor([0,0])

        return prediction_scores, seq_relationship_score

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        #print(type(bert_model_embedding_weights))
        if not config.not_tie_word_embeddings:
            self.decoder.weight = bert_model_embedding_weights
        #print(id(self.decoder.weight))
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    
#
class QBertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(QBertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
      
        
        self.num_qubits = config.n_qubits_MLM
        self.vocab_size = config.vocab_size
        self.num_entangle_circuit_layers = config.n_entangling_circuit_layers_MLM
        self.entangle_params_transform_layer = torch.nn.Linear(config.hidden_size, self.num_entangle_circuit_layers * self.num_qubits)

        # self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.quantum_layer = self.get_circuit()
        self.projection = torch.nn.Linear(2**(self.num_qubits), self.vocab_size)


    def get_circuit(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)
        @qml.qnode(dev, interface='torch')
        def quantum_model(w):
            qml.templates.BasicEntanglerLayers(w, wires=range(self.num_qubits))
            return qml.probs(wires=range(self.num_qubits))
        return quantum_model

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)

        entangle_params = self.entangle_params_transform_layer(hidden_states)
        
        entangle_params = torch.sigmoid(entangle_params.reshape(-1,self.num_entangle_circuit_layers,self.num_qubits))* 2 * math.pi

        qcircuit_output = self.quantum_layer(entangle_params)
        qcircuit_output = qcircuit_output.to(self.projection.weight.dtype)
        #hidden_states = qcircuit_output[:self.vocab_size]
        hidden_states = self.projection(qcircuit_output)
        #hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        #hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    
    
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


    
'''
The next part is the QSANN Head
Here we do some change to the original version, we add MLM task related structure

'''
def generate_value_measurement_code_script(num_qubits,d):
    # get the list of measurements for value
    num_measured_qubit = range(1,num_qubits+1)
    all_measurements = []
    for n in num_measured_qubit:
        combination = list(itertools.combinations(range(num_qubits), n))
        for item in combination:
            all_measurements.append(['X',item])
            all_measurements.append(['Y',item])
            all_measurements.append(['Z',item])
    measurements = all_measurements[:d]

    # get the code scripts
    all_generated_code = []
    for ind, item in enumerate(measurements):
        if item[0] == 'X':
            generated_code = f'qml.PauliX({item[1][0]})'
            for i in item[1][1:]:
                generated_code += f'@qml.PauliX({i})'
            generated_code = f'qml.expval(' + generated_code + f')'
            all_generated_code.append(generated_code)

        elif item[0] == 'Y':
            generated_code = f'qml.PauliY({item[1][0]})'
            for i in item[1][1:]:
                generated_code += f'@qml.PauliY({i})'
            generated_code = f'qml.expval(' + generated_code + f')'
            all_generated_code.append(generated_code)

        elif item[0] == 'Z':
            generated_code = f'qml.PauliZ({item[1][0]})'
            for i in item[1][1:]:
                generated_code += f'@qml.PauliZ({i})'
            generated_code = f'qml.expval(' + generated_code + f')'
            all_generated_code.append(generated_code)

    return all_generated_code


