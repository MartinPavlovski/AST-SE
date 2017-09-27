function base_model = UR_train( Xtrain, Ytrain )

base_model = lasso(Xtrain,Ytrain,'Lambda',0.1);

end

