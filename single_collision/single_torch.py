import os
import torch
import numpy as np

########### simple simulator in pytorch
        
def advance(dt, state, a, imp, x_inc):
    x = state[0]
    v = state[1]
    new_x = torch.zeros_like(x)
    new_v = torch.zeros_like(v)
    for i in range(2):
        new_v[i] = v[i] + imp[i] + a[i] * dt
        new_x[i] = x[i] + dt * new_v[i] + x_inc[i]
    new_state = torch.stack([new_x, new_v], dim=0)
    return new_state

# collision implementation that is easier to understand
def collide_simple(cfg, dt, state, ctrl):
    x = state[0]
    v = state[1]
    imp = torch.zeros((2, 2), dtype=torch.float32)
    x_inc = torch.zeros((2, 2), dtype=torch.float32)

    s_id = 0
    o_id = 1
    v_s_next = v[s_id]+ctrl[s_id]*dt
    v_o_next = v[o_id]+ctrl[o_id]*dt
    penetration_dist = (x[s_id] + v_s_next * dt) - (x[o_id] + v_o_next * dt)
    penetration_dist_norm = torch.sqrt(penetration_dist[0] ** 2 + penetration_dist[1] ** 2)
    if penetration_dist_norm < 2 * cfg.radius:
        penetration_dir = penetration_dist / penetration_dist_norm
        rela_v = v_s_next - v_o_next
        penetration_projected_v = torch.dot(rela_v, penetration_dir)
        if penetration_projected_v < 0.:
            toi = (penetration_dist_norm - 2 * cfg.radius) / torch.min(
                torch.tensor(-1e-3), penetration_projected_v
            )
            # collision direction
            dt_minus_toi = torch.max(dt-toi, torch.tensor(0))
            col_dist = (x[s_id] + v_s_next * dt_minus_toi) - (x[o_id] + v_o_next * dt_minus_toi)
            col_dist_norm = torch.sqrt(col_dist[0] ** 2 + col_dist[1] ** 2)
            col_dir = col_dist / col_dist_norm
            # collision projected velocity
            col_rela_v = v[s_id] + ctrl[s_id] * dt_minus_toi - (v[o_id] + ctrl[o_id] * dt_minus_toi)
            col_projected_v = torch.dot(col_rela_v, col_dir)
            # compute impulse
            impulse = - (1 + 1) * 0.5 * col_projected_v * col_dir
            imp = torch.stack([impulse, -impulse])
            # compute increment in x
            x_inc = - dt_minus_toi * imp

    return imp, x_inc

# collision implementation that follows the style of Taichi
def collide(cfg, dt, state, ctrl):
    x = state[0]
    v = state[1]
    imp = torch.zeros((2, 2), dtype=torch.float32)
    x_inc = torch.zeros((2, 2), dtype=torch.float32)
    for i in range(2):
        s_id = i
        o_id = 1 - i
        v_s_next = v[s_id]+ctrl[s_id]*dt
        v_o_next = v[o_id]+ctrl[o_id]*dt
        penetration_dist = (x[s_id] + v_s_next * dt) - (x[o_id] + v_o_next * dt)
        penetration_dist_norm = torch.sqrt(penetration_dist[0] ** 2 + penetration_dist[1] ** 2)
        if penetration_dist_norm < 2 * cfg.radius:
            penetration_dir = penetration_dist / penetration_dist_norm
            rela_v = v_s_next - v_o_next
            penetration_projected_v = torch.dot(rela_v, penetration_dir)
            if penetration_projected_v < 0.:
                toi = (penetration_dist_norm - 2 * cfg.radius) / torch.min(
                    torch.tensor(-1e-3), penetration_projected_v
                )
                # collision direction
                dt_minus_toi = torch.max(dt-toi, torch.tensor(0))
                col_dist = (x[s_id] + v_s_next * dt_minus_toi) - (x[o_id] + v_o_next * dt_minus_toi)
                col_dist_norm = torch.sqrt(col_dist[0] ** 2 + col_dist[1] ** 2)
                col_dir = col_dist / col_dist_norm
                # collision projected velocity
                col_rela_v = v[s_id] + ctrl[s_id] * dt_minus_toi - (v[o_id] + ctrl[o_id] * dt_minus_toi)
                col_projected_v = torch.dot(col_rela_v, col_dir)
                # compute impulse
                temp_imp = torch.zeros_like(imp)
                temp_imp[s_id] = - (1 + cfg.elasticity) * 0.5 * col_projected_v * col_dir
                # traditionally, the following is calculated
                # temp_imp[s_id] = - (1 + cfg.elasticity) * 0.5 * penetration_projected_v * penetration_dir
                imp = imp + temp_imp
                # compute increment in x
                temp_x_inc = torch.zeros_like(x_inc)
                temp_x_inc[s_id] = torch.min(toi - dt, torch.tensor(0)) * imp[s_id]          
                x_inc = x_inc + temp_x_inc

    return imp, x_inc


def compute_diff_sim_loss(cfg, init_state, ctrls):
    state = init_state
    states_hist = [init_state]
    running_loss = 0
    for i in range(ctrls.shape[0]):
        # imp, x_inc = collide(cfg, cfg.dt, state, ctrls[i])
        # alternative implementation
        imp, x_inc = collide_simple(cfg, cfg.dt, state, ctrls[i])

        next_state = advance(cfg.dt, state, ctrls[i], imp, x_inc)
        state = next_state
        states_hist.append(state)

    x = state[0]
    terminal_loss = x[1][0] ** 2 + x[1][1] ** 2
    running_loss = (ctrls[:, 0] ** 2).sum() * cfg.dt * cfg.epsilon
    loss = terminal_loss + running_loss
    return loss, (states_hist, terminal_loss, running_loss)


if __name__ == "__main__":

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

    init_state = torch.tensor([
        [[cfg.init_pos_0[0], cfg.init_pos_0[1]], [cfg.init_pos_1[0], cfg.init_pos_1[1]]],
        [[0., 0.], [0., 0.]]
    ], dtype=torch.float32, requires_grad=True)

    ctrls = torch.tensor([
        [[cfg.init_const_ctrl[0], cfg.init_const_ctrl[1]], [0., 0.]]
        for _ in range(cfg.steps)
    ], dtype=torch.float32, requires_grad=True)

    states_hist, ctrls_hist, loss_hist = [], [], []
    ctrls_hist.append(np.copy(ctrls.detach().cpu().numpy()[:, 0]))
    for i in range(cfg.train_iters):
        loss, (state_traj, terminal_loss, running_loss) = compute_diff_sim_loss(
            cfg, init_state, ctrls
        )
        loss.backward()
        with torch.no_grad(): # gradient descent update
            ctrls[:, 0] = ctrls[:, 0] - cfg.learning_rate * ctrls.grad[:, 0]
        # zero grad
        ctrls.grad = None
        if cfg.verbose:
            print(f"Iter: {i}, loss: {loss.item()}")
        states_hist.append(torch.stack(state_traj))
        ctrls_hist.append(np.copy(ctrls.detach().cpu().numpy()[:, 0]))
        loss_hist.append(loss.item())

    states_hist = torch.stack(states_hist).detach().cpu().numpy()

    save_dir = os.path.join(cfg.this_dir, cfg.result_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, cfg.name),
        ctrls_hist=np.stack(ctrls_hist),
        states_hist=np.stack(states_hist),
        loss_hist=np.array(loss_hist)
    )