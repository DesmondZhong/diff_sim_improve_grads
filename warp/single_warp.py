import os
import sys
from customized_integrator_euler_direct import CustomizedSymplecticEulerIntegratorDirect
from _single_warp import TwoBalls
import numpy as np


class CONFIGURATION:
    simulation_time = 1.
    steps = 480
    dt = simulation_time / steps
    name = os.path.basename(__file__)[:-3]
    this_dir = os.path.abspath('')
    result_dir = "results"
    learning_rate = 200.
    epsilon = 0.01
    radius = 0.2
    elasticity = 1.0
    init_pos_0 = [-1., -2.]
    init_pos_1 = [-1., -1.]
    init_const_ctrl = [0.0, 3.0]
    target = [0.0, 0.0]
    train_iters = 500
    verbose=True

cfg = CONFIGURATION()

u_init = np.array([[cfg.init_const_ctrl[0], cfg.init_const_ctrl[1]] for _ in range(cfg.steps)])

system = TwoBalls(
    cfg,
    integrator_class=CustomizedSymplecticEulerIntegratorDirect,
    adapter='cpu', 
    render=False,
    u_init=u_init
)
loss = system.compute_loss()
print("------------Gradients-----------")
print(f"loss: {loss}")

x_grad = system.check_grad(system.states[0].particle_q)
v_grad = system.check_grad(system.states[0].particle_qd)
ctrl0_grad = system.check_grad(system.states[0].external_particle_f)
print(f"gradient of loss w.r.t. initial position dl/dx0: {x_grad.numpy()[:, 0:2]}")
print(f"gradient of loss w.r.t. initial velocity dl/dv0: {v_grad.numpy()[:, 0:2]}")
print(f"gradient of loss w.r.t. initial ctrl dl/du0: {ctrl0_grad.numpy()[:, 0:2]}")

system = TwoBalls(
    cfg,
    integrator_class=CustomizedSymplecticEulerIntegratorDirect,
    adapter='cpu', 
    render=False,
    u_init=u_init
)

print("---------start training------------")
loss_np, ctrls_np = system.train()
print("---------finish training------------")
np.savez(
    os.path.join(system.save_dir, cfg.name), 
    loss=loss_np, 
    ctrls=ctrls_np
)
