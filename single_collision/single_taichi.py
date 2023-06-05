import os
import taichi as ti
import numpy as np


class CONFIGURATION:
    simulation_time = 1.
    steps = 480
    dt = simulation_time / steps
    name = os.path.basename(__file__)[:-3]
    this_dir = os.path.abspath('')
    result_dir = "results"
    taichi_dir = "taichi_figures"
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

steps = cfg.steps
epsilon = cfg.epsilon
dt = cfg.dt
learning_rate = cfg.learning_rate
radius = cfg.radius
elasticity = cfg.elasticity

vis_interval = 8
output_vis_interval = 8
vis_resolution = 1024

#######################
# initiate taichi
real = ti.f32
ti.init(kernel_profiler=False, default_fp=real, flatten_if=True)

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()
terminal_loss = scalar()
running_loss = scalar()
x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()
ctrls = vec()

goal = ti.Vector(list(cfg.target))

ti.root.dense(ti.i, steps+1).dense(ti.j, 2).place(x, v, x_inc, impulse)
ti.root.dense(ti.i, steps).dense(ti.j, 2).place(ctrls)
ti.root.place(loss)
ti.root.place(terminal_loss)
ti.root.place(running_loss)
ti.root.lazy_grad()


@ti.kernel
def collide(t: ti.i32):
    for i in range(2):
        s_id = i
        o_id = 1 - i
        imp = ti.Vector([0.0, 0.0])
        x_inc_contrib = ti.Vector([0.0, 0.0])
        v_s_next = v[t, s_id] + ctrls[t, s_id] * dt
        v_o_next = v[t, o_id] + ctrls[t, o_id] * dt
        penetration_dist = (x[t, s_id] + dt * v_s_next) - (x[t, o_id] + dt * v_o_next)
        penetration_dist_norm = penetration_dist.norm()
        if penetration_dist_norm < 2 * radius:
            penetration_dir = penetration_dist / penetration_dist_norm
            penetration_rela_v = v_s_next - v_o_next
            penetration_projected_v = penetration_dir.dot(penetration_rela_v)

            if penetration_projected_v < 0:
                toi = (penetration_dist_norm - 2 * radius) / min(
                    -1e-3, penetration_projected_v
                )
                dt_minus_toi = max(dt-toi, 0.)
                # collision direction
                col_dist = (x[t, s_id] + v_s_next * dt_minus_toi) - (x[t, o_id] + v_o_next * dt_minus_toi)
                col_dir = ti.Vector.normalized(col_dist)
                
                # collision projected velocity
                col_rela_v = v[t, s_id] + ctrls[t, s_id] * dt_minus_toi - (v[t, o_id] + ctrls[t, o_id] * dt_minus_toi)  
                col_projected_v = col_dir.dot(col_rela_v)
                # compute impulse
                imp = - (1 + elasticity) * 0.5 * col_projected_v * col_dir
                # traditionally we would have computed the impulse as follows
                # imp = - (1 + elasticity) * 0.5 * penetration_projected_v * penetration_dir
                # compute increment in x
                x_inc_contrib = - dt_minus_toi * imp
        impulse[t + 1, s_id] += imp
        x_inc[t + 1, s_id] += x_inc_contrib


@ti.kernel
def advance(t: ti.i32):
    for i in range(2):
        v[t, i] = v[t - 1, i] + impulse[t, i] + ctrls[t - 1, i] * dt 
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]


@ti.kernel
def compute_terminal_loss():
    # terminal cost
    loss[None] = (x[steps, 1][0] - goal[0]) ** 2 + (x[steps, 1][1] - goal[1]) ** 2


@ti.kernel
def compute_running_loss():
    # running cost
    for t in range(steps):
        ti.atomic_add(
            loss[None],
            epsilon * (ctrls[t, 0][0] ** 2 + ctrls[t, 0][1] ** 2) * dt
        )


@ti.kernel
def initialize_ctrls():
    for t in range(steps):
        ctrls[t, 0]= [cfg.init_const_ctrl[0], cfg.init_const_ctrl[1]]
        ctrls[t, 1] = [0., 0.]


def fit_to_canvas(p):
    return (p + 2.) / 2.


def forward(cfg, iter=0):

    interval = vis_interval
    pixel_radius = int(radius * 1024) + 1

    for t in range(1, steps + 1):
        collide(t - 1)
        advance(t) # from t - 1 to t
        # print(f"Iter {t}", x[t, 0], x[t, 1], v[t, 0], v[t, 1])
        # print(v[t])
        if (t + 1) % interval == 0 and iter % 100 == 0:
            gui.clear()
            gui.circle((fit_to_canvas(goal[0]), fit_to_canvas(goal[1])), 0x00000, pixel_radius // 2)
            colors = [0xCCCCCC, 0x3344cc]
            for i in range(2):
                gui.circle((fit_to_canvas(x[t, i][0]), fit_to_canvas(x[t, i][1])), colors[i], pixel_radius// 2)
            gui.show(os.path.join(cfg.this_dir, cfg.taichi_dir, f"image_{t}.png"))

    compute_terminal_loss()
    compute_running_loss()


@ti.kernel
def clear():
    for t, i in ti.ndrange(steps + 1, 2):
        impulse[t, i] = ti.Vector([0.0, 0.0])
        x_inc[t, i] = ti.Vector([0.0, 0.0])

@ti.kernel
def step():
    for t in range(steps):
        ctrls[t, 0][0] -= learning_rate * ctrls.grad[t, 0][0]
        ctrls[t, 0][1] -= learning_rate * ctrls.grad[t, 0][1]


def optimize():
    initialize_ctrls()
    # initial condition
    x[0, 0][0] = cfg.init_pos_0[0] ; x[0, 0][1] = cfg.init_pos_0[1]
    x[0, 1][0] = cfg.init_pos_1[0] ; x[0, 1][1] = cfg.init_pos_1[1]

    clear()
    loss_hist, ctrls_hist, states_hist = [], [], []
    for iter in range(cfg.train_iters):
        clear()
        with ti.ad.Tape(loss):
            forward(cfg, iter)
        loss_hist.append(loss[None]) # loss[None] is a python scalar
        if cfg.verbose:
            print('Iter=', iter, 'Loss=', loss[None])
        step()
        ctrls_hist.append(ctrls.to_numpy()[:, 0]) # 480, 2 (balls), 2 (x, y)
        states_hist.append(x.to_numpy())
    clear()
    return {
        'loss_hist': np.array(loss_hist),
        'ctrls_hist': np.stack(ctrls_hist, axis=0),
        'states_hist': np.stack(states_hist, axis=0)
    }


if __name__ == '__main__':
    # render
    try: 
        gui = ti.GUI("Single collision", (1024, 1024), background_color=0x3C733F)
    except RuntimeError: # running on a headless node
        import pyvirtualdisplay

        _display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                            size=(1400, 1400))
        _ = _display.start()
        gui = ti.GUI("Single collision", (1024, 1024), background_color=0x3C733F)
    os.makedirs(os.path.join(cfg.this_dir, cfg.taichi_dir), exist_ok=True)

    # optimize
    return_dict = optimize()

    save_dir = os.path.join(cfg.this_dir, cfg.result_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, cfg.name), 
        **return_dict
    )
    