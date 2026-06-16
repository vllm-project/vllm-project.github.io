#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "open3"
require "date"
require "yaml"

MAX_SUMMARY_CHARS = 240
POSTS_PATHSPEC = ":(glob)_posts/*.md"

def run_git(*args)
  stdout, stderr, status = Open3.capture3("git", *args)
  return stdout if status.success?

  warn "git #{args.join(" ")} failed:"
  warn stderr
  exit 1
end

def github_event_payload
  path = ENV["GITHUB_EVENT_PATH"]
  return {} if path.nil? || path.empty? || !File.file?(path)

  JSON.parse(File.read(path))
rescue JSON::ParserError
  {}
end

def changed_posts_from_github_event
  event_name = ENV["GITHUB_EVENT_NAME"]
  payload = github_event_payload

  range =
    if event_name == "pull_request" && ENV["GITHUB_BASE_REF"] && !ENV["GITHUB_BASE_REF"].empty?
      "origin/#{ENV.fetch("GITHUB_BASE_REF")}...HEAD"
    elsif event_name == "push" && payload["before"] && !payload["before"].match?(/\A0+\z/)
      "#{payload.fetch("before")}..HEAD"
    end

  return [] unless range

  run_git("diff", "--name-only", range, "--", POSTS_PATHSPEC).lines.map(&:strip)
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
  return all_posts if ARGV.delete("--all")
  return ARGV if ARGV.any?

  paths =
    if ENV["GITHUB_ACTIONS"] == "true"
      changed_posts_from_github_event
    else
      changed_posts_from_worktree
    end

  paths.select { |path| path.start_with?("_posts/") && path.end_with?(".md") && File.file?(path) }.uniq.sort
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

  if summary.empty?
    failures << "#{path}: missing `summary` front matter for SEO"
  elsif summary.length > MAX_SUMMARY_CHARS
    failures << "#{path}: `summary` is #{summary.length} characters; maximum is #{MAX_SUMMARY_CHARS}"
  end
end

if failures.any?
  warn "Blog post summary check failed."
  warn "Each changed blog post must include `summary` front matter for SEO."
  warn "Keep it concise, specific, and #{MAX_SUMMARY_CHARS} characters or fewer."
  warn
  failures.each { |failure| warn "- #{failure}" }
  exit 1
end

if posts.empty?
  puts "No changed blog posts found; `summary` front matter for SEO check skipped."
else
  puts "Checked #{posts.length} blog post(s) for `summary` front matter for SEO."
end
