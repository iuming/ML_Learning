#!/usr/bin/env python3
import asyncio
import numpy as np
from caproto.server import pvproperty, pvgroup, ioc_arg_parser, run

# --- your existing imports and sim code go here ---
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


# ---------------------------------------
# General parameters
# ---------------------------------------
# parameters
Ts = 1e-6                   # simulation time step, s

# only for simulation here
t_fill  = 510               # length of cavity filling stage, sample
t_flat  = 1300              # end time of the flattop stage, sample

# ---------------------------------------
# RF source
# ---------------------------------------
# parameters
fsrc = -460                # offset frequency from carrier, Hz
Asrc = 1                    # RF source amplitude, V

# simulate per time step
pha_src = 0                 # state variable
def sim_rfsrc(fsrc, Asrc, pha_src, Ts):
    pha = pha_src + 2.0 * np.pi * fsrc * Ts
    return Asrc*np.exp(1j*pha), pha

# ---------------------------------------
# I/Q modulator
# ---------------------------------------
# parameters
pulsed      = False                                     # pulsed or CW mode
buf_size    = 2048 * 8                                  # buffer size
base_pul    = np.zeros(buf_size, dtype = complex)       # buffer for baseband signal
base_cw     = 1                                         # complex CW baseband scalar input  

base_pul[:t_flat] = 1.0

# simulate per step
buf_id = 0                                              # id to get the buffer data
def sim_iqmod(sig_in, pulsed = True, base_pul = None, base_cw = 0, buf_id = 0):
    if pulsed:
        sig_out = sig_in * base_pul[buf_id if buf_id < len(base_pul) else -1]
    else:
        sig_out = sig_in * base_cw
    return sig_out

# ---------------------------------------
# amplifier
# ---------------------------------------
# parameters
gain_dB = 20*np.log10(12e6)

# simulate
def sim_amp(sig_in, gain_dB):
    return sig_in * 10.0**(gain_dB / 20.0) 

# ---------------------------------------
# cavity
# ---------------------------------------
# parameters
mech_modes = {'f': [280, 341, 460, 487, 618],
              'Q': [40, 20, 50, 80, 100],
              'K': [2, 0.8, 2, 0.6, 0.2]}

f0   = 1.3e9                                 # RF operating frequency, Hz
beta = 1e4
roQ  = 1036                                  # r/Q of the cavity, Ohm
QL   = 3e6                                   # loaded quality factor
RL   = 0.5 * roQ * QL                        # loaded resistance (Linac convention), Ohm
wh   = np.pi * f0 / QL                       # half bandwidth, rad/s
ib   = 0.008                                 # average beam current, A
dw0  = 2*np.pi*000                           # initial detuning, rad/s

beam_pul = np.zeros(buf_size, dtype = complex)  # buffer for pulsed beam 
beam_cw  = 0                                    # complex CW beam
beam_pul[t_fill:t_flat] = ib

# derived parameters
status, Am, Bm, Cm, Dm = cav_ss_mech(mech_modes)
status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm, 
                                     Ts       = Ts, 
                                     method   = 'zoh', 
                                     plot     = False,
                                     plot_pno = 10000)

# simulation
state_m  = np.matrix(np.zeros(Bd.shape))        # state of the mechanical equation
state_vc = 0.0                                  # state of cavity equation

def sim_cav(half_bw, RL, dw_step0, detuning0, vf_step, state_vc, Ts, beta = 1e4,
            state_m0 = 0, Am = None, Bm = None, Cm = None, Dm = None,
            pulsed = True, beam_pul = None, beam_cw = 0, buf_id = 0):
    # get the beam
    if pulsed:
        vb = -RL * beam_pul[buf_id if buf_id < len(beam_pul) else -1]
    else:
        vb = beam_cw

    # execute for one step
    status, vc, vr, dw, state_m = sim_scav_step(half_bw,
                                                dw_step0,
                                                detuning0, 
                                                vf_step, 
                                                vb, 
                                                state_vc, 
                                                Ts, 
                                                beta      = beta,
                                                state_m0  = state_m0, 
                                                Am        = Am, 
                                                Bm        = Bm, 
                                                Cm        = Cm, 
                                                Dm        = Dm,
                                                mech_exe  = True)           
    state_vc = vc
    
    # return 
    return vc, vr, dw, state_vc, state_m

# ---------------------------------------
# RF system simulator
# ---------------------------------------
sim_len = 2048 * 16
pul_len = 2048 * 10

sig_src = np.zeros(sim_len, dtype = complex)
sig_iqm = np.zeros(sim_len, dtype = complex)
sig_amp = np.zeros(sim_len, dtype = complex)
sig_vc  = np.zeros(sim_len, dtype = complex)
sig_vr  = np.zeros(sim_len, dtype = complex)
sig_dw  = np.zeros(sim_len, dtype = complex)

dw = 0

class VirtualCavity(pvgroup.PVGroup):
    # We'll publish real‐time values of |Vc|, ∠Vc, and detuning in Hz
    Vc_amp = pvproperty(value=0.0,       name='VC:AMPL', units='MV',
                        precision=4,       doc='Simulated cavity voltage amplitude')
    Vc_phase = pvproperty(value=0.0,     name='VC:PHAS', units='deg',
                          precision=2,     doc='Simulated cavity phase')
    Detune = pvproperty(value=0.0,       name='VC:DTUN', units='Hz',
                        precision=1,      doc='Simulated cavity detuning')

    async def _run(self):
        """Background task: step the sim and update PVs at ~1 ms intervals."""
        # --- initialize your sim states here ---
        Ts = 1e-3   # slower for demo: 1 ms timestep
        bd, bm, cm, dm = Bd, Bm, Cm, Dm   # discrete mech matrices
        state_vc = 0.0
        state_m = np.zeros((bm.shape[0],))
        dw = 0.0

        while True:
            # one step of your sim: (vc, vr, dw, new_vc_state, new_m_state)
            vc, vr, dw, state_vc, state_m = sim_scav_step(
                half_bw, dw, dw0, vf_step, beam_vb,
                state_vc, Ts, beta, state_m, Am=bd, Bm=bm, Cm=cm, Dm=dm
            )

            # publish
            amp = abs(vc) * 1e-6        # convert to MV
            phase = np.angle(vc) * 180/np.pi
            self.Vc_amp.set(amp)
            self.Vc_phase.set(phase)
            self.Detune.set(dw / (2*np.pi))

            await asyncio.sleep(Ts)

async def main():
    ioc_options, run_options = ioc_arg_parser(
        default_prefix='VIRT_CAV:',
        desc="Virtual RF Cavity Soft IOC")
    ioc = VirtualCavity(**ioc_options)
    await run(ioc.pvdb, **run_options)

if __name__ == '__main__':
    asyncio.run(main())
