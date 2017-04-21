function base_model = base_train( Xtrain, Ytrain )

base_model = lasso(Xtrain,Ytrain,'Lambda',0.1);

end

