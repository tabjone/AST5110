from nm_lib import nm_lib as nm
import numpy as np

class HydroSolver:
    def __init__(self):
        #init
        pass 

    def calculate_drho_dt(self):
        ux = self.ux
        rho = self.rho
        xx = self.xx

        ddx_rho = nm.deriv_upw(xx, rho)[np.where(ux>=0)] + nm.deriv_dnw(xx, rho)[np.where(ux<0)]

        drhodt = -rho * nm.deriv_cent(xx, ux) - ux * ddx_rho
        return drhodt
    
    def calculate_drhoux_dt(self):
        ux = self.ux
        rho = self.rho
        xx = self.xx
        Pg = self.Pg
        Bz = self.Bz
        By = self.By

        ddx_rhoux = nm.deriv_upw(xx, rho * ux)[np.where(ux>=0)] + nm.deriv_dnw(xx, rho * ux)[np.where(ux<0)]

        ddx_ux_x = nm.deriv_upw(xx, ux)[np.where(ux>=0)] + nm.deriv_dnw(xx, ux)[np.where(ux<0)]

        ddx_Pg = nm.deriv_cent(xx, Pg)

        ddx_Bz_Bz = nm.deriv_upw(xx, Bz)[np.where(Bz>=0)] + nm.deriv_dnw(xx, Bz)[np.where(Bz<0)]
        ddx_By_By = nm.deriv_upw(xx, By)[np.where(By>=0)] + nm.deriv_dnw(xx, By)[np.where(By<0)]

        drhoux_dt = - rho * ux * ddx_ux_x + ux * ddx_rhoux - ddx_Pg - Bz * ddx_Bz_Bz - By * ddx_By_By

        return drhoux_dt

    def calculate_drhouy_dt(self):
        uy = self.uy
        rho = self.rho
        xx = self.xx
        ux = self.ux
        Bx = self.Bx
        By = self.By

        ddx_rhouy = nm.deriv_upw(xx, rho * uy)[np.where(uy>=0)] + nm.deriv_dnw(xx, rho * uy)[np.where(uy<0)]
        ddx_ux_y = nm.deriv_upw(xx, ux)[np.where(uy>=0)] + nm.deriv_dnw(xx, ux)[np.where(uy<0)]
        ddx_By_Bx = nm.deriv_upw(xx, By)[np.where(Bx>=0)] + nm.deriv_dnw(xx, By)[np.where(Bx<0)]

        drhouy_dt = - rho * uy * ddx_ux_y + ux * ddx_rhouy + Bx * ddx_By_Bx
    
        return drhouy_dt
    
    def calculate_drhouz_dt(self):
        uz = self.uz
        rho = self.rho
        xx = self.xx
        ux = self.ux
        Bx = self.Bx
        Bz = self.Bz

        ddx_rhouz = nm.deriv_upw(xx, rho * uz)[np.where(uz>=0)] + nm.deriv_dnw(xx, rho * uz)[np.where(uz<0)]
        ddx_ux_z = nm.deriv_upw(xx, ux)[np.where(uz>=0)] + nm.deriv_dnw(xx, ux)[np.where(uz<0)]
        ddx_Bz_Bx = nm.deriv_upw(xx, Bz)[np.where(Bx>=0)] + nm.deriv_dnw(xx, Bz)[np.where(Bx<0)]

        drhouz_dt = - rho * uz * ddx_ux_z + ux * ddx_rhouz - Bx * ddx_Bz_Bx
    
        return drhouz_dt
    
    def calculate_de_dt(self):
        ux = self.ux
        xx = self.xx
        e = self.e
        Pg = self.Pg

        ddx_e_x = np.deriv_upw(xx, e)[np.where(ux>=0)] + np.deriv_dnw(xx, e)[np.where(ux<0)]

        dedt = - ux * ddx_e_x - (e + Pg) * nm.deriv_cent(xx, ux)
        return dedt
    
    def calculate_dBx_dt(self):
        return np.zeros(self.Bx.shape)
    
    def calculate_dBy_dt(self):
        ux = self.ux
        uy = self.uy
        xx = self.xx
        Bx = self.Bx
        By = self.By

        ddx_ux_By = nm.deriv_upw(xx, ux)[np.where(By>=0)] + nm.deriv_dnw(xx, ux)[np.where(By<0)]
        ddx_By_x = nm.deriv_upw(xx, By)[np.where(ux>=0)] + nm.deriv_dnw(xx, By)[np.where(ux<0)]
        ddx_uy_Bx = nm.deriv_upw(xx, uy)[np.where(Bx>=0)] + nm.deriv_dnw(xx, uy)[np.where(Bx<0)]
        ddx_Bx_y = nm.deriv_upw(xx, Bx)[np.where(uy>=0)] + nm.deriv_dnw(xx, Bx)[np.where(uy<0)]

        dBy_dt = - By * ddx_ux_By - ux * ddx_By_x + Bx * ddx_uy_Bx + uy * ddx_Bx_y

        return dBy_dt
    
    def calculate_dBz_dt(self):
        ux = self.ux
        uz = self.uz
        xx = self.xx
        Bx = self.Bx
        Bz = self.Bz

        ddx_ux_Bz = nm.deriv_upw(xx, ux)[np.where(Bz>=0)] + nm.deriv_dnw(xx, ux)[np.where(Bz<0)]
        ddx_Bz_x = nm.deriv_upw(xx, Bz)[np.where(ux>=0)] + nm.deriv_dnw(xx, Bz)[np.where(ux<0)]
        ddx_uz_Bx = nm.deriv_upw(xx, uz)[np.where(Bx>=0)] + nm.deriv_dnw(xx, uz)[np.where(Bx<0)]
        ddx_Bx_z = nm.deriv_upw(xx, Bx)[np.where(uz>=0)] + nm.deriv_dnw(xx, Bx)[np.where(uz<0)]

        dBz_dt = Bz * ddx_ux_Bz + ux * ddx_Bz_x - uz * ddx_Bx_z - Bx * ddx_uz_Bx
        return dBz_dt

    def step(self):
        self.drhodt = self.calculate_drho_dt()
        self.drhouxdt = self.calculate_drhoux_dt()
        self.drhouydt = self.calculate_drhouy_dt()
        self.drhouzdt = self.calculate_drhouz_dt()
        self.dedt = self.calculate_de_dt()
        self.dBxdt = self.calculate_dBx_dt()
        self.dBydt = self.calculate_dBy_dt()
        self.dBzdt = self.calculate_dBz_dt()