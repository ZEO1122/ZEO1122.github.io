#!/usr/bin/env ruby
require "nokogiri"
require "uri"
require "cgi"

root = File.expand_path(ARGV.fetch(0, "_site"))
abort "Build the site first: #{root}" unless File.directory?(root)
files = Dir.glob(File.join(root, "**", "*.html"))
errors = []
documents = {}
files.each { |file| documents[file] = Nokogiri::HTML(File.read(file)) }
documents.each do |file, doc|
  relative = file.delete_prefix(root)
  errors << "#{relative}: Liquid error" if doc.text.include?("Liquid error")
  doc.css("a[href], img[src]").each do |node|
    href = node["href"] || node["src"]
    next if href.nil? || href.empty? || href.match?(/\A(?:mailto:|tel:|javascript:|data:)/)
    begin
      uri = URI.parse(URI::DEFAULT_PARSER.escape(href, /[^\x00-\x7F]/))
      next if uri.host && uri.host != "zeo1122.github.io"
      next if uri.scheme && !%w[http https].include?(uri.scheme)
      decoded = URI::DEFAULT_PARSER.unescape(uri.path.to_s)
      target = if decoded.empty?
        file
      elsif decoded.start_with?("/")
        File.join(root, decoded)
      else
        File.expand_path(decoded, File.dirname(file))
      end
      target = File.join(target, "index.html") if File.directory?(target)
      target += ".html" if !File.exist?(target) && File.file?(target + ".html")
      unless File.file?(target)
        errors << "#{relative}: missing target #{href}"
        next
      end
      if uri.fragment && documents[target]
        id = URI::DEFAULT_PARSER.unescape(uri.fragment)
        found = documents[target].css("[id], a[name]").any? { |n| n["id"] == id || n["name"] == id }
        errors << "#{relative}: missing anchor #{href}" unless id.empty? || found
      end
    rescue URI::InvalidURIError
      errors << "#{relative}: invalid URL #{href}"
    end
  end
end

%w[en ko].each do |lang|
  prefix = lang == "ko" ? "/ko" : ""
  %w[lg-aimers-9th discord-bot transformer-study].each do |slug|
    path = "#{root}#{prefix}/portfolio/#{slug}/index.html"
    errors << "Missing project: #{path}" unless documents[path]
  end
  about = documents["#{root}#{prefix}/index.html"]
  rows = about.css("table tr").select { |row| row.text.match?(/LG Aimers 9(?:기|th)/) }
  errors << "#{lang}: expected one August LG Aimers 9th activity" unless rows.size == 1 && rows.first.text.include?("2026.08") && rows.first.text.include?("63")
  cv = documents["#{root}#{prefix}/cv/index.html"]
  errors << "#{lang}: missing Notion link" unless cv.css('a[href*="3b2b6928b66c81dd95f7e2b9dcde038c"]').size == 1
end

if errors.empty?
  puts "PASS: #{files.size} HTML files; internal links, images, anchors, project pages, activity deduplication, and Notion links."
else
  warn errors.uniq.join("\n")
  abort "#{errors.uniq.size} content errors"
end
