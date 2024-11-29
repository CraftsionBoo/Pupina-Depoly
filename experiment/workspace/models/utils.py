import os 

# 日志输出
class Logger(object):
    def __init__(self):
        self._logger = None
    
    def init(self, logdir, name="log"):
        if self._logger == None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel("INFO")
            fh = logging.FileHandler(log_file)  # 文件保存
            sh = logging.StreamHandler()        # 终端显示
            self._logger.addHandler(fh)
            self._logger.addHandler(sh)
    
    def info(self, str_info):
        self.init("./logs", "tmp.log")
        self._logger.info(str_info)
logger = Logger()
print = logger.info

# 保存为权重
def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))
def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saveing model to {}".format(expand_user(new_file)))
        
    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v 
    torch.save(state_dict, expand_user(new_file))

# onnx导出
def export2onnx(model, weights_root, onnx_file, dummy_input):
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(torch.load(weights_root))
    model.eval()
    torch.onnx.export(
        model, dummy_input, onnx_file, 
        export_params=True, 
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input" : {0 : 'batch_size'}, "output" : {0 : 'bactch_size'}},
        verbose=True
    )
    print("model has benn converted to onnx!\n")