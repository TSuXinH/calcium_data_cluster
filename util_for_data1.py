import h5py

"""
Raw file content:
    CaA0, CaA1: shown below
    
    licks:
    
    summary:
    
    trials:

CaA# content:
    FOV: 200 488
    
    F_dF: the fluorescence traces, an N by 1 cell where N is the number of trials. 
        Each cell contains an M x P double array where M is the number of ROIs and P is the number of timepoints in the n_th trial.
        
    ROIs: profile of the interested cell in the image.
    
    cellid: currently unknown
    
    celltype: 
        the decoded celltypes of the recorded ROIs, an M x 1 double array where the number indicates the decoded cell type of the ROI.
        
    deconv: the deconvolved calcium events, an 1 by N cell where N is the number of trials.
        Each cell contains an M x P double array where M is the number of ROIs and P is the number of timepoints in the n_th trial.
        
    Sampling rate: 32.5868
    
    trial_info: contains trial info necessary to link calcium recording with behavior data. 1 x N cell where N is the number of trials. 
        CaA0.trial_info{n}.mat_file tells you the name of the nth trial which can be used to reference the correct trial in summary table (see summary table below)
    

    trial_info content:
        fileloc: currently unknown
        
        mat_file: currently unknown
        
        motion_metric: currently unknown
        
        time_stamp: currently unknown


licks: 
    lick_vector: long vector
    
    time_stamp: different from the 'time_stamp' above
    
    trial: float number indicating n_th trial, like trial[0] -> 1.

summary:
    CaA0_leave_out:
    
    CaA1_leave_out:
    
    CaA2_leave_out:
    
    bhv_dir:
    
    img_dir:
    
    table:
"""


def fetch_trial_info(c):
    ti_ref = c['trial_info'][:].reshape(-1)
    result = []
    for item in ti_ref:
        tmp_ti = c[item]
        fl = tmp_ti['fileloc'][:]
        mf = tmp_ti['mat_file'][:]
        mm = tmp_ti['motion_metric'][:]
        ts = tmp_ti['time_stamp'][:]
        ti_content = [fl, mf, mm, ts]
        result.append(ti_content)
    return result
