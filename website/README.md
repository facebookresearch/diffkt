# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

Make sure your fork of the repo at Github, and you local repo, have 
the branch gh-pages from the main repo.

### Installation
1. First install nodejs

   https://nodejs.org/en/

2. The Docusaurus website is already intialized but you need to reinstall components that are not tracked by git.

   cd diffkt/website
   npm install

2.1 If you need to reinstall Docusaurus from scratch, the below command will do it but it will 
    overwrite some of the DiffKt website files. After running this command you need to revert all the changed
    files back to the last commit.

   cd diffkt/website
   npx create-docusaurus@latest my-website facebook
   git add . --dry-run           To see the files that were changed
   git checkout -- <file_name>             For each file changed

3. Install Latex
   cd diffkt/website
   npm install --save remark-math@3 rehype-katex@5 hast-util-is-element@1.1.0

### Local Development
   cd diffkt/website
   npm run start

### Build
    cd diffkt/website
    npm run build

### Deployment
    To be on the safe side:
    	1. sync your fork of origin/gh-pages to upstream/gh-page, in case
    	   facebook Open Source support directly added something to gh-pages
    	2. cd diffkt
    	3. locally, git checkout gh-pages
    	4. locally, git pull
    	5. locally, git checkout <branch with docusaurus>
    	
    cd diffkt/website
    ./publish.sh

    After running the script, the diffkt/website/build directory will 
    be the gh-pages branch of your fork locally and on Github. You can
    then create a PR to merge it to the main repo.

