import unittest

from hydrasim.core.reactor import (
    BoundaryConditions,
    IntegratedCSTR,
    ReactorConfiguration,
)


class TestReactorRealism(unittest.TestCase):
    def test_chloramine_formation_from_ammonia(self):
        cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_chlorine=2.5,
            initial_chloramine=0.0,
            initial_ammonia=0.8,
            initial_chlorine_demand=0.0,
            temperature=20.0,
            n_zones=5,
        )
        reactor = IntegratedCSTR(cfg)
        boundary = BoundaryConditions(
            inlet_flow_rate=0.0, acid_flow_rate=0.0, chlorine_flow_rate=0.0
        )

        cl0 = float(reactor.state.chlorine[0])
        nh30 = float(reactor.state.ammonia[0])
        cl_comb0 = float(reactor.state.chloramine[0])

        for _ in range(120):
            reactor.step(1.0, boundary)

        cl1 = float(reactor.state.chlorine[0])
        nh31 = float(reactor.state.ammonia[0])
        cl_comb1 = float(reactor.state.chloramine[0])

        self.assertLess(cl1, cl0)
        self.assertLess(nh31, nh30)
        self.assertGreater(cl_comb1, cl_comb0)

    def test_bulk_chlorine_demand_reduces_free_chlorine(self):
        cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_chlorine=2.0,
            initial_chloramine=0.0,
            initial_ammonia=0.0,
            initial_chlorine_demand=2.5,
            temperature=20.0,
            n_zones=5,
        )
        reactor = IntegratedCSTR(cfg)
        boundary = BoundaryConditions(
            inlet_flow_rate=0.0, acid_flow_rate=0.0, chlorine_flow_rate=0.0
        )

        cl0 = float(reactor.state.chlorine[0])
        demand0 = float(reactor.state.chlorine_demand[0])

        for _ in range(120):
            reactor.step(1.0, boundary)

        cl1 = float(reactor.state.chlorine[0])
        demand1 = float(reactor.state.chlorine_demand[0])

        self.assertLess(cl1, cl0)
        self.assertLess(demand1, demand0)

    def test_dosing_streams_affect_hydraulics_and_physics(self):
        cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_pH=7.2,
            initial_chlorine=0.1,
            initial_ammonia=0.0,
            initial_chlorine_demand=0.0,
            n_zones=5,
        )
        reactor = IntegratedCSTR(cfg)
        boundary = BoundaryConditions(
            inlet_flow_rate=0.0,
            acid_flow_rate=0.8,
            acid_concentration=0.5,
            chlorine_flow_rate=0.4,
            chlorine_concentration=80.0,
            inlet_chlorine=0.0,
        )

        pH0 = float(reactor.state.pH[0])
        cl0 = float(reactor.state.chlorine[0])

        for _ in range(30):
            reactor.step(1.0, boundary)

        pH1 = float(reactor.state.pH[0])
        cl1 = float(reactor.state.chlorine[0])

        self.assertGreater(reactor.state.flow_rate, 1.1)
        self.assertLess(pH1, pH0)
        self.assertGreater(cl1, cl0)

    def test_chloramine_formation_has_expected_ph_dependence(self):
        low_ph_cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_pH=6.0,
            initial_chlorine=2.0,
            initial_chloramine=0.0,
            initial_ammonia=1.0,
            initial_chlorine_demand=0.0,
            temperature=20.0,
            n_zones=5,
        )
        mid_ph_cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_pH=8.4,
            initial_chlorine=2.0,
            initial_chloramine=0.0,
            initial_ammonia=1.0,
            initial_chlorine_demand=0.0,
            temperature=20.0,
            n_zones=5,
        )
        high_ph_cfg = ReactorConfiguration(
            flow_rate=0.0,
            initial_pH=10.0,
            initial_chlorine=2.0,
            initial_chloramine=0.0,
            initial_ammonia=1.0,
            initial_chlorine_demand=0.0,
            temperature=20.0,
            n_zones=5,
        )
        boundary = BoundaryConditions(
            inlet_flow_rate=0.0, acid_flow_rate=0.0, chlorine_flow_rate=0.0
        )

        low_ph_reactor = IntegratedCSTR(low_ph_cfg)
        mid_ph_reactor = IntegratedCSTR(mid_ph_cfg)
        high_ph_reactor = IntegratedCSTR(high_ph_cfg)

        for _ in range(120):
            low_ph_reactor.step(1.0, boundary)
            mid_ph_reactor.step(1.0, boundary)
            high_ph_reactor.step(1.0, boundary)

        self.assertGreater(
            float(mid_ph_reactor.state.chloramine[0]),
            float(low_ph_reactor.state.chloramine[0]),
        )
        self.assertGreater(
            float(mid_ph_reactor.state.chloramine[0]),
            float(high_ph_reactor.state.chloramine[0]),
        )

    def test_through_flow_advects_chlorine_downstream(self):
        cfg = ReactorConfiguration(
            volume=1000.0,
            n_zones=5,
            flow_rate=100.0,
            impeller_speed=10.0,
            initial_pH=7.2,
            initial_chlorine=0.0,
            initial_chloramine=0.0,
            initial_ammonia=0.0,
            initial_chlorine_demand=0.0,
            temperature=20.0,
            inlet_chlorine=2.5,
            inlet_ammonia=0.0,
            inlet_chlorine_demand=0.0,
        )
        reactor = IntegratedCSTR(cfg)
        boundary = BoundaryConditions(
            inlet_flow_rate=100.0,
            inlet_pH=7.2,
            inlet_chlorine=2.5,
            inlet_ammonia=0.0,
            inlet_chlorine_demand=0.0,
            inlet_temperature=20.0,
            acid_flow_rate=0.0,
            chlorine_flow_rate=0.0,
        )

        for _ in range(60):
            reactor.step(10.0, boundary)

        self.assertGreater(float(reactor.state.chlorine[2]), 0.2)
        self.assertGreater(float(reactor.state.chlorine[-1]), 0.05)
        self.assertGreater(
            float(reactor.state.chlorine[0]), float(reactor.state.chlorine[-1])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
