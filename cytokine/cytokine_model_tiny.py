#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


# In[72]:


class Cytokine1(object):
    
    def __init__(self,**kwargs):
        '''
        mu_pro = 1/20, # rate of clearance
        c0     = 10,   # rate of  background production
        c1     = 10,   # 
        c2     = 50,   #
        n1     = 2, 
        c3     = 10,   #(1/hr)
        c4     = 40,   #(pg/ml)
        n2     =  2,
        c5     = 10,   # (1/hr)
        I50    = 40,
        mu_ant = 1/5,  #(1/hr)
        c6     = 10,   #(pg/ml)
        c7     = 5,    #1/hr
        n3     = 2,
        c8     = 30,   #(1/hr)
        K50    = 40,   # (pg/ml)
        mu_plasma = 1/8, # 1/hr
        a0 = 0,
        s0 = 20,# initial dosage (pg/ml)
        p0 = 0  # initial dosage (pg/ml)
        '''
        
        self.params = kwargs.copy()
        
        self.ignore_anti_inflammatory = self.params.pop("ignore_anti_inflammatory",True)
        
        p0 = self.params.pop("p0")
        s0 = self.params.pop("s0")
        
        if self.ignore_anti_inflammatory:
            x0 = np.array([p0,s0])
        else:
            a0 = self.params.pop("a0")
            x0 = np.array([p0,a0,s0])
        
        self.initial_values = x0
        self.current_values = x0
        self.last_evaluated_time = 0.0 # time zero when consent time
        self.step_size  = 1/60.0    # 1 minutes evaluation
        
    def p_positive_feedback(self, p,c0=10,c1=10,c2=50,n1=2,**kwargs):
        # pro-inflammator concetration (pg/ml)

        # The background pro-inflammatory production rate
        # c0    background production rate "(pg/ml/hr)" 
        # c1    maximum production rate "(pg/ml/hr)"
        # c2    concentration where the half of maximum is secreted (pg/ml)

        # n1  cytokine receptor expression

        return c0*(1 + c1*(p**n1)/(c2**n1 + p**n1))
    
    def p_inhibition(self, a, c3,c4,n2=2,**kwargs):

        # c3   The regulation strength if anti-inflammatory cytokines acting on pro-inflammatory cytokines
        # c4   The regulation strength if anti-inflammatory cytokines acting on pro-inflammatory cytokines
        # n2   Cyto4kine receptor expression

        return c3*(c4**n2)/(c4**n2+a**n2)
    
    
    def p_steriod(self, s,c5,I50,**kwargs): 
        # c5   Efficiency of steroid on inhibiting pro-inflammatory 
        # I50  The concentration of steroid which binds to the half of the receptors
        return c5*(I50/(I50+s))
    
    def pro_inflammatory_model(self, p,a, s, mu_pro, c0,c1,c2,n1,I50,c5=1, **kwargs):
        if self.ignore_anti_inflammatory:
            dp = - mu_pro*p + self.p_positive_feedback(p,c0,c1,c2,n1)*self.p_steriod(s,c5,I50)
        else:
            c3 = kwargs["c3"]
            c4 = kwargs["c4"]
            n2 = kwargs["n2"]
            dp = - mu_pro*p + self.p_positive_feedback(p,c0,c1,c2,n1)*self.p_inhibition(a, c3,c4,n2)*self.p_steriod(s,c5,I50)
        return dp
    
    def a_upregulate(self, p, c6, c7, n3, **kwargs):
        return c7*((p**n3)/(c6**n3 + p**n3))
    
    def a_steriod(self, s, c8, K50, **kwargs):
        return c8*(K50/(K50+s))

    def anti_inflammatory_model(self, p, a, s, mu_ant, c6, c7, n3, c8, K50, **kwargs):
        da = -mu_ant*a + self.a_upregulate(p, c6, c7, n3)*self.a_steriod(s, c8, K50)
        return da
    
    def steriod(self,s,t, mu_plasma, **kwargs):
        ds = -mu_plasma*s
        return ds

    
    def model(self, x, t ):
        
        if self.ignore_anti_inflammatory:
            (p,s) = x
            dx = [self.pro_inflammatory_model(p,np.nan,s,**self.params),
                  self.steriod(s,t,**self.params)]
        else:
            (p,a,s) = x
            dx = [self.pro_inflammatory_model(p,a,s,**self.params),
                  self.anti_inflammatory_model(p,a,s,**self.params),
                  self.steriod(s,t,**self.params)]
        return dx

    
    def evaluate_model(self, t, initial_values=None):
        """definition of function for LS fit
            x gives evaluation points,
            teta is an array of parameters to be varied for fit"""
        
        if initial_values is None:
            initial_values = self.initial_values
        
        # create an alias to f which passes the optional params    
        model_with_params = lambda x,t: self.model(x, t)
        # calculate ode solution, retuen values for each entry of "x"
        r = odeint(model_with_params,initial_values,t)
        #in this case, we only need one of the dependent variable values
        return r
    
    def step_to(self, time):
        old_values = self.current_values
        if self.last_evaluated_time<time:
            t =  np.arange(self.last_evaluated_time, time, self.step_size)
            initial_values = self.current_values
            out = self.evaluate_model( t, initial_values)
            self.current_values = out[-1,:]
            self.last_evaluated_time = time
            
        return self.current_values,old_values
    
    def reset(self):
        self.current_values = self.initial_values
        self.last_evaluated_time = 0.0


# In[79]:


if __name__=="__main__":
    
    model1 = Cytokine1(        mu_pro = 1/20, # rate of clearance
            c0     = 10,   # rate of  background production
            c1     = 10,   # 
            c2     = 50,   #
            n1     = 2, 
            c3     = 10,   #(1/hr)
            c4     = 40,   #(pg/ml)
            n2     =  2,
            c5     = 10,   # (1/hr)
            I50    = 40,
            mu_ant = 1/5,  #(1/hr)
            c6     = 10,   #(pg/ml)
            c7     = 5,    #1/hr
            n3     = 2,
            c8     = 30,   #(1/hr)
            K50    = 40,   # (pg/ml)
            mu_plasma = 1/8, # 1/hr
            a0 = 0,
            s0 = 40,# initial dosage (pg/ml)
            p0 = 0  # initial dosage (pg/ml))
            )
    mm = []
    for t in range(0,30):
        o = model1.step_to(t)
        mm.append(o[2])
    plt.plot(mm)

