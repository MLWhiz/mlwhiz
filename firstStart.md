When Starting the repo. 

You need to get hugo: 0.82.0 Version
just run brew install hugo.rb

Also need to work with submodule. 
rm -rf ~/personal/mlwhiz/public

Remove the submodule from Git configuration:

git submodule deinit -f public

Remove the submodule entry from .git/config and .gitmodules:

git rm -f public

Clean up any remaining submodule-related files:

rm -rf .git/modules/public

Re-add the submodule:

git submodule add -b master git@github.com:MLWhiz/mlwhiz.github.io.git public

Initialize and update:

git submodule update --init --recursive public

finally run sh fulldeploy.sh