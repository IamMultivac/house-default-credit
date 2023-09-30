space_column = "sk_id_curr"
target_column = "target"
prediction_column = "prediction"


base_learners_params = {
     "boosting_type": 'gbdt',
     "n_estimators": 1000,
     "num_leaves": 39,
     "min_child_samples": 94,
     "subsample": 0.9674035250863153,
     "learning_rate": 0.0088112031800569,
     "colsample_bytree": 0.9750067130759722,
     "lambda_l1": 8.865861216071197,
     "lambda_l2": 0.02873006473756534,
     "n_jobs":-1,
     "random_state": 42,
     "verbose":-1
}


boruta_learner_params = {'boosting_type': 'gbdt',
 'n_estimators': 550,
 'num_leaves': 39,
 'min_child_samples': 94,
 'subsample': 0.9674035250863153,
 'learning_rate': 0.01,
 'colsample_bytree': 0.9750067130759722,
 'lambda_l1': 8.865861216071197,
 'lambda_l2': 0.02873006473756534,
 'n_jobs': -1,
 'random_state': 42,
 'verbose': -1
                        }

test_params = {
	"learner_params": {
		"learning_rate": 0.0088112031800569,
		"n_estimators": 2090,
		"extra_params": {
			'objective': "binary",
			'metric': "binary_logloss",
			'boosting_type': 'gbdt',
			'num_leaves': 39,
			'min_child_samples': 94,
			'subsample': 0.9674035250863153,
			'colsample_bytree': 0.9750067130759722,
			'lambda_l1': 8.865861216071197,
			'lambda_l2': 0.02873006473756534,
			'n_jobs': -1,
			'random_state': 42,
            "monotone_constraints":None,
			"verbose": -1
		}
	}
}

MODEL_PARAMS = {
	'learner_params': {
		'n_estimators': 5926, 
        'learning_rate': 0.005603627873630697,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'gbdt',
			 'lambda_l1': 0.00021744689137046032,
             'lambda_l2': 6.07402119317552,
             'num_leaves': 31,
             'feature_fraction': 0.4,
             'bagging_fraction': 0.7762748139696756,
             'bagging_freq': 7,
             'min_child_samples': 20,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': None,
			'verbose': -1
		}
	}
}


params_optuna = {
	'learner_params': {
		'learning_rate': 0.005603627873630697,
		'n_estimators': 5926,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'gbdt',
			'lambda_l1': 3.201016964067897e-07,
             'lambda_l2': 4.0276123590778266,
             'num_leaves': 200,
             'feature_fraction': 0.48000000000000004,
             'bagging_fraction': 0.8100007780166677,
             'bagging_freq': 3,
             'min_child_samples': 100,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': None,
			'verbose': -1
		}
	}
}

params_ensemble = {
	'learner_params': {
		'learning_rate': 0.005603627873630697,
		'n_estimators': 5926,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'gbdt',
			 'feature_pre_filter': False,
             'lambda_l1': 3.778715480993499e-08,
             'lambda_l2': 9.701943176300759,
             'num_leaves': 124,
             'feature_fraction': 0.41600000000000004,
             'bagging_fraction': 0.7401877344450646,
             'bagging_freq': 2,
             'min_child_samples': 100,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': None,
			'verbose': -1
		}
	}
}

params_fw = {
	'learner_params': {
		'learning_rate': 0.005603627873630697,
		'n_estimators': 5926,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'gbdt',
			 'lambda_l1': 0.17381521305946443,
             'lambda_l2': 8.821433034912198,
             'num_leaves': 111,
             'feature_fraction': 0.4,
             'bagging_fraction': 0.7396714409529886,
             'bagging_freq': 6,
             'min_child_samples': 20,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': None,
			'verbose': -1
		}
	}
}

params_all = {
	'learner_params': {
		'learning_rate': 0.005603627873630697,
		'n_estimators': 5926,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'gbdt',
			 'lambda_l1': 1.0073717100534047e-08,
             'lambda_l2': 8.932762924966607,
             'num_leaves': 150,
             'feature_fraction': 0.5,
             'bagging_fraction': 0.6580775817581002,
             'bagging_freq': 7,
             'min_child_samples': 20,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': None,
			'verbose': -1
		}
	}
}

