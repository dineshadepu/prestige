"""100 spheres falling inside hopper
Check the complete molecular dynamics code
"""
from __future__ import print_function
import numpy as np

# PybPH base and carray imports
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.equation import Group
from pysph.solver.application import Application
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from pysph.examples.rigid_body.ten_spheres_in_vessel_2d import get_fluid_and_dam_geometry


def get_particle_array_dem(constants=None, **props):
    dem_props = ['x0', 'y0', 'u0', 'v0', 'fx', 'fy', 'fz', 'R']

    pa = get_particle_array(constants=constants, additional_props=dem_props,
                            **props)

    pa.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'pid', 'tag', 'gid', 'fx', 'fy'])

    return pa


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0):
        self.gx = gx
        self.gy = gy
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy):
        d_fx[d_idx] = d_m[d_idx] * self.gx
        d_fy[d_idx] = d_m[d_idx] * self.gy


class LinearSpringForceParticleParticle(Equation):
    """Documentation for LinearSpringForce
    """

    def __init__(self, dest, sources, kn=1e4):
        super(LinearSpringForceParticleParticle, self).__init__(dest, sources)
        self.kn = kn

    def loop(self, d_idx, d_m, d_fx, d_fy, VIJ, XIJ, RIJ, d_R, s_idx, s_R):
        overlap = 0

        if RIJ > 0:
            overlap = d_R[d_idx] + s_R[s_idx] - RIJ

        if overlap > 0:
            # basic variables: normal vector
            _rij = 1. / RIJ
            nx = -XIJ[0] * _rij
            ny = -XIJ[1] * _rij
            d_fx[d_idx] += self.kn * overlap * nx
            d_fy[d_idx] += self.kn * overlap * ny


class DEMStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_x, d_y, d_u0, d_v0, d_u, d_v):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]

    def stage1(self, d_idx, d_x0, d_y0, d_x, d_y, d_u0, d_v0, d_fx, d_fy, d_u,
               d_v, d_m, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_fx[d_idx] / d_m[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_fy[d_idx] / d_m[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_x, d_y, d_u0, d_v0, d_fx, d_fy, d_u,
               d_v, d_m, dt):
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]

        d_u[d_idx] = d_u0[d_idx] + dt * d_fx[d_idx] / d_m[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_fy[d_idx] / d_m[d_idx]


class FluidStructureInteration(Application):
    def initialize(self):
        # self.spacing = 0.1
        self.spacing = 0.01
        # self.spacing = 0.0109
        # self.spacing = 0.00324
        # self.spacing = 0.001

    def create_particles(self):
        spcg = self.spacing
        (xt, yt, xb, yb) = get_fluid_and_dam_geometry(4., 5., 1., 1., 1, spcg,
                                                      spcg)
        # setup the particles as per example
        xt -= 1.
        yt -= 1.
        xb -= 1.5
        yb -= 0.6

        r = spcg / 2.
        m = 2000. * spcg**2.
        body = get_particle_array_dem(x=xb, y=yb, m=m, R=r, h=r, name="body")

        tank = get_particle_array_dem(x=xt, y=yt, m=m, R=r, h=r, name="tank")
        print("spacing is ", spcg)
        print("Body particles", len(body.x))
        print("Tank particles", len(tank.x))
        print("number of particles ", len(body.x) + len(tank.x))
        return [body, tank]

    def create_solver(self):
        kernel = CubicSpline(dim=2)

        integrator = EPECIntegrator(body=DEMStep())

        dt = 1e-4
        print("DT: %s" % dt)
        tf = 10. * dt
        solver = Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt,
                        tf=tf, adaptive_timestep=False, pfreq=1e5)

        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                BodyForce(dest='body', sources=None, gy=-9.81),
                LinearSpringForceParticleParticle(
                    dest='body', sources=['body', 'tank'], kn=1e4),
            ]),
        ]
        return equations


if __name__ == '__main__':
    app = FluidStructureInteration()
    app.run()
