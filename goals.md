
# Goals

| Goal 	     / Feature                                              | Description 	                                                                           | Priority	 | Status	       | 	   |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------|---------------|-----|
| 	       Collect 499 r2 scores                                     | Run a bulk analysis of all 499 files	                                                   | 	High     | In Progress   | 	   |
| 	       Write evaluation application                              | 	Application will read in all r2 scores to plot distribution and other important metric | 	High     | In Progress   | 	   |
| 	       Clone Repo on Exacloud                                    | Setup the repo in exacloud and make sure it is working	                                 | 	High     | Done          | 	   |
| 	       Use full feature files                                    | Full feature files include 40k columns	                                                 | 	Medium   | Listed	       | 	   |
| 	       Adjust model to make use of more layer                    | Models needs to be adjusted for the full feature files	                                 | 	Medium   | Listed	       | 	   |
| 	       Add command to specify which kind of model should be used | Full feature vs Not full feature should use different models	                           | 	Medium   | Done	         | 	   |
| 	       MlFlow Integration                                        | MLFlow integration should be implemented if possible                                    | 	Low      | Listed        | 	   |
| 	       Multi-encoder devel                                       | Make a 2 encoder version, with 256 and 512 batch sizes                                  | 	Medium   | Listed        | 	   |
| 	       Add cross-validation                                      | Set repo to write function only once                                                    | 	Medium   | Listed        | 	   |
| 	       Convert to classification                                 | Sigmoid function, or similar                                                            | 	Low      | Not-started   | 	   |
| 	       Result file aggregation                                   | Can do locally to start, automate eventually                                            | 	Low      | Not-started   | 	   |
| 	       Test number of layers                                     | Create versions with 5, 8, 12 layers                                                    | 	Medium   | Not-started   | 	   |
| 	       Input file generation automation                          | Based on script in source directory                                                     | 	High     | In Progress   | 	   |
| 	       Cross validation                                          | Decide number of splits, relates to number of layers                                    | 	Medium   | Pending data  | 	   |
| 	       Hyperparameter python script                              | Need to achieve parallization                                                           | 	High     | Listed        | 	   |
| 	       Save model as h5 file                                     | Standard format for cancer drug response models                                         | 	High     | To be started | 	   |