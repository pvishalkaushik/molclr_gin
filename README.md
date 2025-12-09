# molclr_gin

git clone https://github.com/pvishalkaushik/molclr_gin.git      
            
cd molclr_gin         
pyenv install 3.7.16       
pyenv virtualenv 3.7.16 molclr_gin_env       
pyenv local molclr_gin_env
pip install torch==1.7.1 torchvision==0.8.2
pip install -r requirements.txt      
          
            

python predict.py --task [task name] --smiles [smiles string]      


# Currently, allowed task names are: bbbp
