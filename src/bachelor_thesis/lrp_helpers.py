class ConservationChecker:
    def __init__(self, module_name):
        self.module_name = module_name
        self.rin = None
        self.rout = None

    def hook(self, module, grad_input, grad_output):
        # grad_output is a tuple of gradients. We are interested in the first one.
        if grad_output[0] is not None:
            self.rin = grad_output[0].sum().item()
        
        # grad_input is also a tuple.
        if grad_input[0] is not None:
            self.rout = grad_input[0].sum().item()
        else:
            # For the very first layer, grad_input might be None
            self.rout = self.rin 

        if self.rin is not None and self.rout is not None:
            diff = self.rin - self.rout
            if not torch.isclose(torch.tensor(self.rin), torch.tensor(self.rout)):
                 print(
                    f"Conservation violation in {self.module_name}: "
                    f"R_in={self.rin:.6f}, R_out={self.rout:.6f}, Diff={diff:.6f}"
                )