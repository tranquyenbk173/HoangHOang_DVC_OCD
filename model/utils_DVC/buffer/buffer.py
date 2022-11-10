from model.utils_DVC.setup_elements import input_size_match
from model.utils_DVC import name_match #import update_methods, retrieve_methods
from model.utils_DVC.utils import maybe_cuda
import torch
from model.utils_DVC.buffer.buffer_utils import BufferClassTracker
from model.utils_DVC.setup_elements import n_classes

class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = True #self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        # define buffer
        buffer_size = params.n_memories
        print('buffer has %d slots' % buffer_size)
        # input_size = input_size_match[params.data]
        input_size = [3, 32, 32]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)

        # define update and retrieve method
        self.update_method = name_match.update_methods['random'](params)
        self.retrieve_method = name_match.retrieve_methods['MGI'](params)

        # if self.params.buffer_tracker:
        #     self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y,**kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)