# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# https://github.com/NVIDIA/warp/blob/main/LICENSE.md

# This file is modified from 
# https://github.com/NVIDIA/warp/blob/main/warp/sim/integrator_euler.py

import warp as wp

@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        dt: float,
                        impulse: wp.array(dtype=wp.vec3),
                        x_inc: wp.array(dtype=wp.vec3),
                        x_new: wp.array(dtype=wp.vec3),
                        v_new: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + impulse[tid] + (f0 * inv_mass) *dt
    x1 = x0 + v1 * dt + x_inc[tid]

    x_new[tid] = x1
    v_new[tid] = v1


@wp.kernel
def collide(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    ctrls: wp.array(dtype=wp.vec3),
    dt: float,
    radius: float,
    impulse: wp.array(dtype=wp.vec3),
    x_inc: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    if tid > 1:
        return 
    s_id = tid          # self id
    o_id = 1 - tid      # other id
    x_s = particle_x[s_id]
    x_o = particle_x[o_id]
    v_s = particle_v[s_id] 
    v_o = particle_v[o_id]
    v_s_next = particle_v[s_id] + ctrls[s_id] * dt
    v_o_next = particle_v[o_id] + ctrls[o_id] * dt
    penetration_dist = (x_s + dt * v_s_next) - (x_o + dt * v_o_next)
    penetration_dist_norm = wp.length(penetration_dist)
    if penetration_dist_norm < 2. * radius:
        penetration_dir = penetration_dist / penetration_dist_norm
        penetration_rela_v = v_s_next - v_o_next
        penetration_projected_v = wp.dot(penetration_dir, penetration_rela_v)

        if penetration_projected_v < 0.:
            toi = (penetration_dist_norm - 2. * radius) / min(
                -1e-3, penetration_projected_v
            )
            dt_minus_toi = max(dt-toi, 0.)
            # collision direction
            col_dist = (x_s + v_s_next * dt_minus_toi) - (x_o + v_o_next * dt_minus_toi)
            col_dir = wp.normalize(col_dist)

            # collision projected velocity
            col_rela_v = v_s + ctrls[s_id] * dt_minus_toi - (v_o + ctrls[o_id] * dt_minus_toi)

            col_projected_v = wp.dot(col_dir, col_rela_v)
            # compute impulse
            imp = - (1. + 1.) * 0.5 * col_projected_v * col_dir
            # traditionally we would have computed the impulse as follows
            # imp = - (1. + 1.) * 0.5 * penetration_projected_v * penetration_dir
            # compute increment in x
            x_inc_contrib = - dt_minus_toi * imp
    wp.atomic_add(impulse, s_id, imp)
    wp.atomic_add(x_inc, s_id, x_inc_contrib)


class CustomizedSymplecticEulerIntegratorDirect:

    def __init__(self):
        pass


    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):
            #----------------------------
            # integrate particles

            if (model.particle_count):

                wp.launch(
                    kernel=collide,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        state_in.external_particle_f,
                        dt, 
                        model.radius,
                    ],
                    outputs=[
                        state_in.impulse,
                        state_in.x_inc
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q, 
                        state_in.particle_qd,
                        state_in.external_particle_f,
                        model.particle_inv_mass, 
                        dt,
                        state_in.impulse,
                        state_in.x_inc,
                    ],
                    outputs=[
                        state_out.particle_q, 
                        state_out.particle_qd
                        ],
                    device=model.device)

            return state_out