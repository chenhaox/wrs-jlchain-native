"""
Drop-in JLChain variant accelerated by native backend implementations.

The original WRS jlchain module remains the pure Python reference. This class
subclasses it and replaces only the runtime kinematics state with native-owned
buffers after finalization.
"""

import numpy as np

import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as py_jlc
from .backends import create_backend


class _BackendLinkProxy:
    def __init__(self, source, owner, joint_id):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_joint_id", joint_id)

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __setattr__(self, name, value):
        if name in ("_source", "_owner", "_joint_id"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._source, name, value)

    @property
    def gl_pos(self):
        return self._owner._kinematics_backend.gl_pos_q[self._joint_id]

    @property
    def gl_rotmat(self):
        return self._owner._kinematics_backend.gl_rotmat_q[self._joint_id]

    @property
    def gl_homomat(self):
        return rm.homomat_from_posrot(self.gl_pos, self.gl_rotmat)

    @property
    def cmodel(self):
        cmodel = self._source._cmodel
        if cmodel is not None:
            cmodel.pose = (self.gl_pos, self.gl_rotmat)
        return cmodel

    @cmodel.setter
    def cmodel(self, cmodel):
        self._source._cmodel = cmodel
        if cmodel is not None:
            cmodel.pose = (self.gl_pos, self.gl_rotmat)

    def install_onto(self, pos=np.zeros(3), rotmat=np.eye(3)):
        return None


class _BackendJointProxy:
    def __init__(self, source, owner, joint_id):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_joint_id", joint_id)
        object.__setattr__(self, "_lnk_proxy", _BackendLinkProxy(source.lnk, owner, joint_id))

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __setattr__(self, name, value):
        if name in ("_source", "_owner", "_joint_id", "_lnk_proxy"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._source, name, value)

    @property
    def motion_value(self):
        return float(self._owner._kinematics_backend.motion_values[self._joint_id])

    @property
    def gl_pos_0(self):
        return self._owner._kinematics_backend.gl_pos_0[self._joint_id]

    @property
    def gl_rotmat_0(self):
        return self._owner._kinematics_backend.gl_rotmat_0[self._joint_id]

    @property
    def gl_homomat_0(self):
        return rm.homomat_from_posrot(self.gl_pos_0, self.gl_rotmat_0)

    @property
    def gl_motion_ax(self):
        return self._owner._kinematics_backend.gl_motion_axis[self._joint_id]

    @property
    def gl_pos_q(self):
        return self._owner._kinematics_backend.gl_pos_q[self._joint_id]

    @property
    def gl_rotmat_q(self):
        return self._owner._kinematics_backend.gl_rotmat_q[self._joint_id]

    @property
    def gl_homomat_q(self):
        return rm.homomat_from_posrot(self.gl_pos_q, self.gl_rotmat_q)

    @property
    def lnk(self):
        return self._lnk_proxy

    @lnk.setter
    def lnk(self, value):
        self._source.lnk = value
        object.__setattr__(self, "_lnk_proxy", _BackendLinkProxy(value, self._owner, self._joint_id))

    def set_motion_value(self, motion_value):
        jnt_values = self._owner.get_jnt_values()
        jnt_values[self._joint_id] = motion_value
        self._owner.fk(jnt_values=jnt_values, update=True)

    def update_globals(self, pos=np.zeros(3), rotmat=np.eye(3), motion_value=0):
        self.set_motion_value(motion_value)


class JLChain(py_jlc.JLChain):
    def __init__(self, *args, backend_name="kdl", **kwargs):
        super().__init__(*args, **kwargs)
        self._backend_name = backend_name
        self._kinematics_backend = None
        self._python_jnts = self.jnts

    def finalize(self,
                 ik_solver=None,
                 identifier_str="test",
                 backend_name=None,
                 backend_max_iterations=100,
                 backend_ik_eps=1e-6,
                 **kwargs):
        super().finalize(ik_solver=ik_solver, identifier_str=identifier_str, **kwargs)
        self._kinematics_backend = create_backend(
            self,
            backend_name=backend_name or self._backend_name,
            max_iterations=backend_max_iterations,
            ik_eps=backend_ik_eps,
        )
        if self._kinematics_backend is None:
            return
        self._python_jnts = self.jnts
        self.jnts = [_BackendJointProxy(jnt, self, i) for i, jnt in enumerate(self._python_jnts)]
        self.fk(jnt_values=self.home, update=True)

    @property
    def backend_name(self):
        if self._kinematics_backend is None:
            return None
        return self._kinematics_backend.name

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        if self._kinematics_backend is not None and update:
            self._gl_flange_pos, self._gl_flange_rotmat = self._kinematics_backend.update_state(jnt_values)
            if toggle_jacobian:
                _, _, j_mat = self._kinematics_backend.fk(jnt_values=jnt_values, toggle_jacobian=True)
                return self._gl_flange_pos, self._gl_flange_rotmat, j_mat
            return self._gl_flange_pos, self._gl_flange_rotmat
        if self._kinematics_backend is not None:
            return self._kinematics_backend.fk(jnt_values=jnt_values,
                                               toggle_jacobian=toggle_jacobian)
        return super().fk(jnt_values=jnt_values,
                          toggle_jacobian=toggle_jacobian,
                          update=update)

    def jacobian(self, jnt_values=None):
        if self._kinematics_backend is not None:
            return self._kinematics_backend.jacobian(jnt_values=jnt_values)
        return super().jacobian(jnt_values=jnt_values)

    @py_jlc.JLChain.assert_finalize_decorator
    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        if self._kinematics_backend is not None:
            if seed_jnt_values is None:
                seed_jnt_values = self.get_jnt_values()
            jnt_values = self._kinematics_backend.ik(tgt_pos=tgt_pos,
                                                     tgt_rotmat=tgt_rotmat,
                                                     seed_jnt_values=seed_jnt_values)
            if jnt_values is not None:
                if option[0] == "m":
                    return [jnt_values]
                return np.asarray(jnt_values)
            if self._ik_solver is None:
                return None
        return super().ik(tgt_pos=tgt_pos,
                          tgt_rotmat=tgt_rotmat,
                          seed_jnt_values=seed_jnt_values,
                          option=option,
                          toggle_dbg=toggle_dbg)
