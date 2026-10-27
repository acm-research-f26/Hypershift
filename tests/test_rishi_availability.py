import unittest
import numpy as np
from rishi_availability import common_screen,membership_age


class AvailabilityTests(unittest.TestCase):
    def test_common_screen_never_uses_test_label_availability(self):
        data=dict(mask=np.array([[1,1,1,0],[1,1,1,1]],bool),momentum=np.array([[np.nan,1,1,1],[1,1,1,1]]),
            reversal=np.ones((2,4)),labelmask=np.zeros((2,4),bool),y=np.zeros((2,4)))
        counts=np.array([[40,29,30,40],[40,40,40,40]])
        before=common_screen(data,np.arange(2),counts)
        np.testing.assert_array_equal(before,[[0,0,1,0],[1,1,1,1]])
        data['labelmask'][:]=True;data['y'][:]=999.
        np.testing.assert_array_equal(common_screen(data,np.arange(2),counts),before)

    def test_membership_age_resets_and_cannot_see_future(self):
        members=np.array([[0,0],[1,0],[1,1],[0,1],[1,1]],bool)
        actual=membership_age(members)
        np.testing.assert_array_equal(actual,[[0,0],[1,0],[2,1],[0,2],[1,3]])
        changed=members.copy();changed[3:]=False
        np.testing.assert_array_equal(membership_age(changed)[:3],actual[:3])


if __name__=='__main__':unittest.main()
