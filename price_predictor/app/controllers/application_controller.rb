class ApplicationController < ActionController::Base
  # Protect from CSRF
  protect_from_forgery with: :exception

  # Make flash available across redirects
  before_action :set_cache_headers

  private

  # Prevent browser from caching sensitive pages
  def set_cache_headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
  end
end
