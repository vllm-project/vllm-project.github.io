#!/usr/bin/env ruby
# frozen_string_literal: true

require "open3"
require "date"
require "yaml"

MAX_SUMMARY_CHARS = 240
IMAGE_EXTENSIONS = %w[.avif .gif .jpeg .jpg .png .svg .webp].freeze

def run_git(*args)
  stdout, stderr, status = Open3.capture3("git", *args)
  return stdout if status.success?

  warn "git #{args.join(" ")} failed:"
  warn stderr
  exit 1
end

def changed_posts_from_worktree
  run_git("status", "--porcelain", "--", "_posts").lines.map do |line|
    path = line[3..].to_s.strip
    path.include?(" -> ") ? path.split(" -> ", 2).last : path
  end
end

def all_posts
  Dir["_posts/*.md"].sort
end

def selected_posts
  return all_posts if ARGV.delete("--all") || ENV["GITHUB_ACTIONS"] == "true"
  return ARGV if ARGV.any?

  changed_posts_from_worktree.select do |path|
    path.start_with?("_posts/") && path.end_with?(".md") && File.file?(path)
  end.uniq.sort
end

def front_matter(path)
  text = File.read(path)
  match = text.match(/\A---\s*\n(.*?)\n---\s*(?:\n|\z)/m)
  return [nil, "missing YAML front matter"] unless match

  data = YAML.safe_load(match[1], permitted_classes: [Date, Time], aliases: true)
  [data || {}, nil]
rescue Psych::SyntaxError => e
  [nil, "invalid YAML front matter: #{e.message.lines.first&.strip}"]
end

failures = []
posts = selected_posts

posts.each do |path|
  metadata, error = front_matter(path)

  if error
    failures << "#{path}: #{error}"
    next
  end

  summary = metadata.fetch("summary", "").to_s.strip
  image = metadata.fetch("image", "").to_s.strip

  if summary.empty?
    failures << "#{path}: missing `summary` front matter for SEO"
  elsif summary.length > MAX_SUMMARY_CHARS
    failures << "#{path}: `summary` is #{summary.length} characters; maximum is #{MAX_SUMMARY_CHARS}"
  end

  if image.empty?
    failures << "#{path}: missing `image` front matter for the custom hero image"
  elsif !image.start_with?("/assets/")
    failures << "#{path}: `image` must reference a repository asset under `/assets/`"
  else
    image_path = image.delete_prefix("/").split(/[?#]/, 2).first
    if !IMAGE_EXTENSIONS.include?(File.extname(image_path).downcase)
      failures << "#{path}: `image` must reference a supported image file"
    elsif !File.file?(image_path)
      failures << "#{path}: `image` asset does not exist: #{image}"
    end
  end
end

if failures.any?
  warn "Blog post metadata check failed."
  warn
  warn "Each blog post must include `summary` and `image` fields in the YAML front matter at the top of the Markdown file."
  warn "Add it between the opening `---` lines, for example:"
  warn
  warn "---"
  warn "layout: post"
  warn 'title: "..."'
  warn 'author: "..."'
  warn 'summary: "How vLLM serves ExampleModel with FP8 KV cache on NVIDIA GPUs for lower-latency long-context inference."'
  warn 'image: /assets/figures/example-model/hero.png'
  warn "---"
  warn
  warn "Write `summary` as a concise SEO description that names the main vLLM feature, model, release, hardware/backend, or deployment problem covered."
  warn "Keep it #{MAX_SUMMARY_CHARS} characters or fewer."
  warn "Set `image` to a custom hero image committed under `/assets/`; the referenced file must exist."
  warn
  failures.each { |failure| warn "- #{failure}" }
  exit 1
end

if posts.empty?
  puts "No changed blog posts found; blog post metadata check skipped."
else
  puts "Checked #{posts.length} blog post(s) for required summary and hero image metadata."
end
