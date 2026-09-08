# SPI identified parameters — X1 pelvis + motor model

Data: 48 clips (walk_diag)

## Result

| quantity | nominal | identified (raw) | exported (clamped) |
|---|---|---|---|
| mass [kg] | 4.3042 | 3.2832 | 3.2832 |
| com [m] | [0.00252285, -0.00063439, 0.03023409] | [0.053954, 0.032057, 0.016958] | [0.053954, 0.032057, 0.016958] |
| I diag [kg m^2] | [0.0268, 0.0108, 0.0218] | [0.057387, 0.126544, 0.167573] | [0.057387, 0.126544, 0.167573] |
| motor kappa | (see config nominal) | {'hip_pitch': 123.36748157287701, 'hip_rolleyaw': 39.254425542914134, 'knee': 112.39666911095277, 'ankle': 11.906666179601725} | same |
| kappa_s | 1.0 | 0.3768 | same |

Multi-step prediction cost: nominal **6.0** -> best **3.1** (2.0x lower).


## Notes

* no clamping applied (--no-clamp or in-box values)
* weak observability without mocap: com_y/z and inertia absorb model error; kappas are all in-box and well identified.

## Optimization history (tail)

  - 47404776.04598391
  - 4.017034271912646
  - 4.830846320889135
  - 3.9625905827699928
  - 4.711438824038018