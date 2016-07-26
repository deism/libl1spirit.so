

/* Options common to all gateways. Meant to be #included */

			if (!strcmp (name, "time") || !strcmp (name, "timing"))
				timing = true;
			if (!strcmp (name, "verbose"))
				verbose = true;
			if (!strcmp (name, "pocs_2fix")){
				if (val && !strcmp(val, "no")){
					pocs_2fix = false;
				}
				else{
					pocs_2fix = true;
				}
			}

			if (!strcmp (name, "calib_lsqr")){
				calib_lsqr = true;
				if (val && !strcmp(val, "no")){
					calib_lsqr = false;
					forbid_calib_lsqr = true;
				}
			}
			if (!strcmp (name, "calib_lsqr_ne")){
				calib_lsqr_ne = true;
				if (val && !strcmp(val, "no")){
					calib_lsqr_ne = false;
				}
			}

			if (!val)
				continue;

			if (!strcmp (name, "lambda_calib"))
				lambda_calib = atof(val);
			if (!strcmp (name, "iters_calib"))
				iters_calib = atoi(val);
			if (!strcmp (name, "calib_fe"))
				calib_freqs = atoi(val);
			if (!strcmp (name, "ksize")){
				char *s = strdup (val), **ss = &s, *tok;
				int i = 0;
				while (i < 3 && (tok = strsep (ss, "x"))){
					ksize[i++] = atoi(tok);
				}
				free (s);
			}

			if (!strcmp (name, "lambda_l1"))
				lambda_l1 = atof(val);
			if (!strcmp (name, "iters_l1"))
				n_iter_l1 = atoi(val);

			if (!strcmp (name, "gpus") || !strcmp (name, "gpu"))
				max_gpus = atoi(val);
