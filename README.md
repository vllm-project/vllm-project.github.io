# vLLM blog source

## Local development

1. Install `jekyll` and `bundler` by running `gem install jekyll bundler`. Ask ChatGPT for help if you encounter any issues.
2. `bundle install` to install the necessary gems.
3. `rm ./Gemfile.lock` if you meet gem error.
5. `bundle exec jekyll server` to compile and start the server.

To add a new blogpost, please refer to `_posts/2023-06-20-vllm.md` as an example. Some notes:
- Google Doc can be saved as markdown format which will make your life easier.
- Note that the blogpost will only show when its date is in the past.
- Put figures under `assets/figures/yourblogname/`.
- Make a pull request.

The blog is automatically built and deployed by GitHub Actions when `main` is pushed to.

## Theme customization

The theme we are using is [Minima](https://github.com/jekyll/minima). If you need to customise anything from this theme, see [Overriding theme defaults](https://jekyllrb.com/docs/themes/#overriding-theme-defaults).