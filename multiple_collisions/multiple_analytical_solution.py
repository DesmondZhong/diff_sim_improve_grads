import numpy as np


def get_analytical_solution(cfg):
    ts = np.linspace(0, 1, 481)[:-1]

    q10 = np.array([cfg.init_pos_0[0], (cfg.wall_y - cfg.radius - cfg.init_pos_0[1]) * 2 + cfg.init_pos_0[1]])
    q20 = np.array([cfg.init_pos_1[0], (cfg.wall_y - cfg.radius - cfg.init_pos_1[1]) * 2 + cfg.init_pos_1[1]])
    T = cfg.simulation_time
    eps = cfg.epsilon

    def batch_dot(a, b):
        return (a * b).sum(axis=1, keepdims=True)

    def get_batch_cost(inp):
        s = inp[:, 0:1]
        angle = inp[:, 1:2]
        n = np.concatenate([np.cos(angle), np.sin(angle)], axis=1) # bs, 2
        n_perp = np.concatenate([-np.sin(angle), np.cos(angle)], axis=1)
        d = q20 - q10 - (0.2 * 2) * n  # bs, 2
        v_prime_dot_n = - (s * s * (T - s) * batch_dot(q20, n) - 6 * eps * batch_dot(d, n)) / (
            4 * eps * s + s * s * (T - s) * (T - s)
        ) # bs, 
        v_prime_dot_n_perp = 3 * batch_dot(d, n_perp) / (2 * s)
        first_term = q20 + (T - s) * v_prime_dot_n * n
        coef_n = batch_dot(d, n) # bs, 
        coef_n_perp = batch_dot(d, n_perp) # bs, 
        v_prime_dot_d = coef_n * v_prime_dot_n + coef_n_perp * v_prime_dot_n_perp # bs
        second_term = eps * (
            + 12 * batch_dot(d, d) / (s ** 3) 
            + 4 / s * (batch_dot(v_prime_dot_n, v_prime_dot_n) + batch_dot(v_prime_dot_n_perp, v_prime_dot_n_perp))
            - 12 / (s * s) * v_prime_dot_d
        )
        cost = batch_dot(first_term, first_term) + second_term
        return cost, (n, n_perp, d, v_prime_dot_n, v_prime_dot_n_perp)

    s = np.linspace(0.01, 0.99, 100)
    angle = np.linspace(np.pi +0.01, 2 * np.pi - 0.01, 100)
    S, ANGLE = np.meshgrid(s, angle)
    S = S.reshape(-1)
    ANGLE = ANGLE.reshape(-1)
    inp = np.stack([S, ANGLE], axis=1) # bs, 2

    batched_cost, aux_data = get_batch_cost(inp)

    idx = np.argmin(batched_cost)

    s = S[idx]
    d = aux_data[2][idx]
    v_prime = aux_data[3][idx] * aux_data[0][idx] + aux_data[4][idx] * aux_data[1][idx]
    t_ = ts[ts < s]
    u_ = (6 * d / (s * s) - 2 * v_prime / s).reshape(-1, 2) + (6 * v_prime / (s * s) - 12 * d / (s ** 3)) * t_.reshape(-1, 1)

    ### need to flip the y direction since we use the mirror method to solve the analytical solution
    u_[:, 1] = - u_[:, 1]
    t_rest = ts[ts>=s]
    u_rest = np.zeros((t_rest.shape[0], 2))
    u_opt = np.concatenate((u_, u_rest), axis=0)
    return batched_cost[idx], u_opt


if __name__ == "__main__":

    import os

    class CONFIGURATION:
        simulation_time = 1.
        steps = 480
        dt = simulation_time / steps
        name = os.path.basename(__file__)[:-3]
        this_dir = os.path.abspath('')
        result_dir = "results"
        learning_rate = 50.
        epsilon = 0.01
        radius = 0.2
        elasticity = 1.0
        init_pos_0 = [0.25, -0.3]
        init_pos_1 = [-0.5, 0.6]
        init_const_ctrl = [-3.5, 3.0]
        target = [0.0, 0.0]
        wall_y = 1.0
        train_iters = 1500
        verbose=True

    cfg = CONFIGURATION()

    cost, u_opt = get_analytical_solution(cfg)
    print("analytical optimal loss: ", cost)