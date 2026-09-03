# SPI identified parameters — X1 pelvis + motor model

Data: None clips (walk_diag)

## Result

| quantity | nominal | identified (raw) | exported (clamped) |
|---|---|---|---|
| mass [kg] | 4.3042 | 3.1520 | 3.1520 |
| com [m] | [0.00252285, -0.00063439, 0.03023409] | [0.022555, 0.024094, -0.01261] | [0.022555, 0.024094, -0.01261] |
| I diag [kg m^2] | [0.0268, 0.0108, 0.0218] | [0.019006, 0.110257, 0.111269] | [0.019006, 0.110257, 0.111269] |
| motor kappa | (see config nominal) | {'hip_pitch': 76.05733927598162, 'hip_rolleyaw': 39.75707053778301, 'knee': 73.06738339710581, 'ankle': 19.763074805608767} | same |
| kappa_s | 1.0 | 0.3701 | same |

Multi-step prediction cost: see params provenance (costs.val nominal=None best=None).


## Notes

* no clamping applied (--no-clamp or in-box values)
* weak observability without mocap: com_y/z and inertia absorb model error; kappas are all in-box and well identified.

## Optimization history (tail)

(none — params loaded from committed results)
