We report the analysis of simples Gaussians using RealNVP in the SW-JKO scheme in "SWGF_RealNVP.ipynb", and using a JKO-ICNN in "JKO_ICNN_Gaussians.ipynb". We also provide the results obtained on the Aggregation equation for the SW-JKO scheme in "SWGF_MLP_Aggregation.ipynb" and for the JKO-ICNN scheme in "JKO_ICNN_Aggregation.ipynb".

The code to reproduce quantitative experiments (Figure 1,8 in the paper) is in the folder "Fokker_Planck_Gaussians". You first need to generate the data by running "python generate_datas.py".
Then, for each experiment, you need to run "main.py" with the right parameters. For example, to get the results of figure 8, for SW-JKO, we run
- python main.py --ntry=15 --ts 0.5 0.9 --tau=0.05 --fixed_t
We provide some batch commands in files "run.sh".
