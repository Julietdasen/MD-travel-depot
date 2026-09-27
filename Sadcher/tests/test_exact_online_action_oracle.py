import unittest
from baselines.md_oracle_types import ResidualMDState, ResidualOracleStatus, enumerate_forced_batches
from baselines.exact_online_action_oracle import CompleteOnlineAction, enumerate_complete_online_actions, solve_exact_action_continuation
from simulation_environment.domain_model import MaterialDeliveryConfig, ProcessRobot, ProcessTask, SchedulingDomain, TransportRobot, TransportTask

class ExactOnlineActionTests(unittest.TestCase):
    def state(self, domain, pending):
        return ResidualMDState(0,{r.robot_id:r.location for r in domain.robots},(),tuple(pending))
    def test_parallel_vs_idle_q(self):
        d=SchedulingDomain.create(config=MaterialDeliveryConfig(enabled=True),tasks=(ProcessTask(1,(0,0),1,(True,)),ProcessTask(2,(0,0),1,(True,))),robots=(ProcessRobot(0,(0,0),(True,),1),ProcessRobot(1,(0,0),(True,),1)))
        s=self.state(d,(1,2)); actions=enumerate_complete_online_actions(d,s)
        idle=next(a for a in actions if len(a.assignments)==1)
        parallel=next(a for a in actions if len(a.assignments)==2)
        self.assertEqual(solve_exact_action_continuation(d,s,idle).continuation_cost,2.0)
        self.assertEqual(solve_exact_action_continuation(d,s,parallel).continuation_cost,1.0)
    def test_coalition_is_complete_and_redundant_rejected(self):
        d=SchedulingDomain.create(config=MaterialDeliveryConfig(enabled=True),tasks=(ProcessTask(1,(0,0),1,(True,True)),),robots=(ProcessRobot(0,(0,0),(True,False),1),ProcessRobot(1,(0,0),(False,True),1)))
        a=enumerate_complete_online_actions(d,self.state(d,(1,)))
        self.assertEqual(len(a),1); self.assertEqual(a[0].assignments,((0,1),(1,1)))
    def test_all_idle_and_duplicate_robot_rejected(self):
        with self.assertRaises(ValueError): CompleteOnlineAction((),(0,))
        with self.assertRaises(ValueError): CompleteOnlineAction(((0,1),(0,2)),())
    def test_transport_is_singleton(self):
        d=SchedulingDomain.create(config=MaterialDeliveryConfig(enabled=True),tasks=(ProcessTask(2,(0,0),1,(True,)),TransportTask(1,(2,0),(0,0),1,1,1,2),),robots=(ProcessRobot(2,(0,0),(True,),1),TransportRobot(0,(0,0),True,1,1,1),TransportRobot(1,(0,0),True,1,1,1)), material_edges=((1,2),))
        a=enumerate_complete_online_actions(d,self.state(d,(1,2)))
        self.assertEqual({len(x.assignments) for x in a},{1})

if __name__=='__main__': unittest.main()
