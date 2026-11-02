"""Regression tests for native link poses; requires WRS and the KDL extension.

Set WRS_SOURCE to the directory containing the WRS Python package if needed.
Run: python -m unittest discover -s tests -p test_link_poses.py -v
"""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
if os.environ.get("WRS_SOURCE"):
    sys.path.insert(0, os.environ["WRS_SOURCE"])

import wrs
from wrs.basis import robot_math as rm
from wrs.modeling import collision_model as mcm
from wrs.robot_sim._kinematics.collision_checker import CollisionChecker
from wrs.robot_sim._kinematics.jlchain import JLChain as PythonJLChain
from wrs_jlchain_native.jlchain import JLChain as NativeJLChain


def make_chain(chain_type):
    chain = chain_type(pos=np.array([.2, -.1, .3]),
                       rotmat=rm.rotmat_from_euler(.1, .2, .3), n_dof=2)
    for i, joint in enumerate(chain.jnts):
        joint.loc_pos = np.array([.05, .02, .2 * i])
        joint.loc_rotmat = rm.rotmat_from_euler(.2 * i, .1, 0)
        joint.loc_motion_ax = np.array([0., 0., 1.])
        joint.lnk.loc_pos = np.array([.12, -.03, .07])
        joint.lnk.loc_rotmat = rm.rotmat_from_euler(.3, -.2, .4)
        joint.lnk.cmodel = mcm.gen_box(xyz_lengths=np.array([.02, .03, .04]))
    chain.finalize()
    return chain


class LinkPoseTests(unittest.TestCase):
    def test_link_and_mesh_poses_match_python_at_multiple_configurations(self):
        native, reference = make_chain(NativeJLChain), make_chain(PythonJLChain)
        self.assertIn(native.backend_name, ("kdl", "orocos_kdl"))
        for q in ([0, 0], [.4, -.7], [-.5, .9]):
            native.goto_given_conf(q)
            reference.goto_given_conf(q)
            for actual, expected in zip(native.jnts, reference.jnts):
                np.testing.assert_allclose(actual.gl_pos_q, expected.gl_pos_q, atol=1e-12)
                np.testing.assert_allclose(actual.lnk.gl_pos, expected.lnk.gl_pos, atol=1e-12)
                np.testing.assert_allclose(actual.lnk.gl_rotmat, expected.lnk.gl_rotmat, atol=1e-12)
                np.testing.assert_allclose(actual.lnk.cmodel.pos, expected.lnk.cmodel.pos, atol=1e-12)
                np.testing.assert_allclose(actual.lnk.cmodel.rotmat, expected.lnk.cmodel.rotmat, atol=1e-12)

    def test_local_pose_edits_and_saved_meshes(self):
        native, reference = make_chain(NativeJLChain), make_chain(PythonJLChain)
        for chain in (native, reference):
            chain.jnts[0].lnk.loc_pos = np.array([.25, .08, -.1])
            chain.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(-.5, .7, .2)
            chain.goto_given_conf([.2, -.6])
        link = native.jnts[0].lnk
        np.testing.assert_allclose(link.gl_homomat, reference.jnts[0].lnk.gl_homomat, atol=1e-12)
        snapshot = mcm.CollisionModel(link.cmodel)
        expected_pos, expected_rotmat = snapshot.pos.copy(), snapshot.rotmat.copy()
        native.goto_given_conf([.8, .4])
        np.testing.assert_array_equal(snapshot.pos, expected_pos)
        np.testing.assert_array_equal(snapshot.rotmat, expected_rotmat)

    def test_collision_uses_link_offset_not_joint_origin(self):
        chain = make_chain(NativeJLChain)
        chain.goto_given_conf([.4, -.7])
        joint = chain.jnts[0]
        expected_pos = joint.gl_pos_q + joint.gl_rotmat_q @ joint.lnk.loc_pos
        checker = CollisionChecker("link_offset_regression")
        element_id = checker.add_cce(joint.lnk)
        checker.enable_extcd_by_id_list([element_id], type="from")
        obstacle = mcm.gen_sphere(pos=expected_pos, radius=.01)
        self.assertTrue(checker.is_collided(obstacle_list=[obstacle]))
        obstacle.pos = joint.gl_pos_q.copy()
        self.assertFalse(checker.is_collided(obstacle_list=[obstacle]))


if __name__ == "__main__":
    unittest.main()
