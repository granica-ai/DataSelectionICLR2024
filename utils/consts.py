VALID_LABEL_METHODS = ['logistic', 'misspec', 'misspec2']
VALID_INTEGRATION_METHODS = ['GH', 'Trapez', 'TrapezExtrap']
EXPERIMENT_KEYS = ["mis_error_test", "mis_error_val", "log_loss_test", "log_loss_val",
                   "acc_train", "theta_surr_norm", "grad_norm", "alpha_0", "alpha_s", "alpha_perp",
                   "effectiveN"
                   ]

PARAMETER_KEYS = ['beta_0', 'beta_perp', "surr_score", 'N_gen', 'lambd', "alpha_exp",
                  'data_p', 'split_pct', "lambd_surr"]
