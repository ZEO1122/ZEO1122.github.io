#!/usr/bin/env ruby

require "date"
require "yaml"

root = File.expand_path("..", __dir__)
data_path = File.join(root, "_data", "cv.yml")
cv = YAML.safe_load(File.read(data_path), permitted_classes: [Date], aliases: true)
errors = []

walk = lambda do |value, path|
  case value
  when Hash
    localized = value.key?("en") || value.key?("ko")
    if localized
      %w[en ko].each do |locale|
        errors << "#{path}: missing #{locale}" unless value.key?(locale)
        errors << "#{path}.#{locale}: blank value" if value[locale].respond_to?(:empty?) && value[locale].empty?
      end
    end

    value.each { |key, child| walk.call(child, "#{path}.#{key}") }
  when Array
    value.each_with_index { |child, index| walk.call(child, "#{path}[#{index}]") }
  end
end

walk.call(cv, "cv")

%w[_pages/cv.md _pages/cv-ko.md _includes/cv-content.html _layouts/cv.html].each do |relative_path|
  errors << "missing file: #{relative_path}" unless File.file?(File.join(root, relative_path))
end

if errors.empty?
  puts "CV locale check passed."
  exit 0
end

warn errors.join("\n")
exit 1
