import os

import numpy as np

import wrs.robot_sim._kinematics.constant as const


class NativeKDLBackend:
    def __init__(self, jlc, max_iterations=100, ik_eps=1e-6):
        import _wrs_jlchain_native_kdl

        joint_types = np.asarray(
            [2 if jnt.type == const.JntType.PRISMATIC else 1 for jnt in jlc.jnts],
            dtype=np.int32,
        )
        loc_pos = np.asarray([jnt.loc_pos for jnt in jlc.jnts], dtype=float).reshape((jlc.n_dof, 3))
        loc_rotmat = np.asarray([jnt.loc_rotmat for jnt in jlc.jnts], dtype=float).reshape((jlc.n_dof, 3, 3))
        loc_motion_axis = np.asarray([jnt.loc_motion_ax for jnt in jlc.jnts], dtype=float).reshape((jlc.n_dof, 3))
        self._backend = _wrs_jlchain_native_kdl.KDLBackend(
            anchor_pos=np.asarray(jlc.anchor.gl_flange_pose_list[0][0], dtype=float),
            anchor_rotmat=np.asarray(jlc.anchor.gl_flange_pose_list[0][1], dtype=float),
            loc_pos=loc_pos,
            loc_rotmat=loc_rotmat,
            loc_motion_axis=loc_motion_axis,
            joint_types=joint_types,
            joint_ranges=np.asarray(jlc.jnt_ranges, dtype=float),
            flange_joint_id=int(jlc.flange_jnt_id),
            loc_flange_pos=np.asarray(jlc.loc_flange_pos, dtype=float),
            loc_flange_rotmat=np.asarray(jlc.loc_flange_rotmat, dtype=float),
            max_iterations=max_iterations,
            ik_eps=ik_eps,
        )
        self.motion_values = self._backend.motion_values_view()
        self.gl_pos_0 = self._backend.gl_pos_0_view()
        self.gl_rotmat_0 = self._backend.gl_rotmat_0_view()
        self.gl_motion_axis = self._backend.gl_motion_axis_view()
        self.gl_pos_q = self._backend.gl_pos_q_view()
        self.gl_rotmat_q = self._backend.gl_rotmat_q_view()
        self.gl_flange_pos = self._backend.gl_flange_pos_view()
        self.gl_flange_rotmat = self._backend.gl_flange_rotmat_view()

    @property
    def name(self):
        return self._backend.name

    def fk(self, jnt_values, toggle_jacobian=False):
        return self._backend.fk(np.asarray(jnt_values, dtype=float), bool(toggle_jacobian))

    def jacobian(self, jnt_values=None):
        if jnt_values is None:
            jnt_values = self.motion_values
        return self.fk(jnt_values, toggle_jacobian=True)[2]

    def update_state(self, jnt_values):
        self._backend.update_state(np.asarray(jnt_values, dtype=float))
        return self.gl_flange_pos, self.gl_flange_rotmat

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values):
        return self._backend.ik(
            np.asarray(tgt_pos, dtype=float),
            np.asarray(tgt_rotmat, dtype=float),
            np.asarray(seed_jnt_values, dtype=float),
        )


def create_backend(jlc, backend_name=None, **kwargs):
    mode = backend_name or os.environ.get("WRS_JLCHAIN_BACKEND", "auto")
    mode = mode.lower()
    if mode in ("python", "none", "off", "false", "0"):
        return None
    if mode not in ("auto", "kdl", "orocos", "orocos_kdl"):
        raise ValueError(f"Unsupported JLChain backend: {backend_name}")
    try:
        return NativeKDLBackend(jlc, **kwargs)
    except Exception:
        if mode == "auto":
            return None
        raise
