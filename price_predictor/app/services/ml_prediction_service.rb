require "httparty"
require "json"

# MlPredictionService
#
# Thin HTTP client that calls the Python Flask ML microservice.
# All network / service errors are wrapped into typed exceptions so
# the controller can give the user a meaningful message.
#
# Environment variables:
#   ML_SERVICE_URL  – base URL of Flask service (default: http://localhost:5001)
#
class MlPredictionService
  # Raised when the Flask service is unreachable or returns 503
  ServiceUnavailableError = Class.new(StandardError)

  # Raised when the service is up but the prediction itself fails (4xx)
  PredictionError = Class.new(StandardError)

  ML_SERVICE_URL  = ENV.fetch("ML_SERVICE_URL", "http://localhost:5001").freeze
  CONNECT_TIMEOUT = 5   # seconds — for health / info calls
  READ_TIMEOUT    = 30  # seconds — for prediction (can be slower first time)

  # ----------------------------------------------------------------
  # Public API
  # ----------------------------------------------------------------

  # POST /predict
  # Returns the parsed JSON hash from the Flask service.
  def predict(title:, description: "", rating: nil, review_count: nil, category: "unknown", price: nil)
    raise PredictionError, "Product title is required" if title.to_s.strip.empty?

    body = build_body(title:, description:, rating:, review_count:, category:, price:)

    response = HTTParty.post(
      "#{ML_SERVICE_URL}/predict",
      body:    body.to_json,
      headers: { "Content-Type" => "application/json", "Accept" => "application/json" },
      timeout: READ_TIMEOUT
    )

    handle_response(response)
  rescue HTTParty::Error, Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED,
         SocketError, Errno::EHOSTUNREACH => e
    raise ServiceUnavailableError,
          "Cannot reach ML service at #{ML_SERVICE_URL} — #{e.class}: #{e.message}"
  end

  # GET /health
  # Returns a hash; never raises — callers check the "status" key.
  def health_check
    response = HTTParty.get(
      "#{ML_SERVICE_URL}/health",
      timeout: CONNECT_TIMEOUT,
      headers: { "Accept" => "application/json" }
    )
    JSON.parse(response.body)
  rescue StandardError => e
    { "status" => "error", "message" => e.message }
  end

  # GET /models
  # Returns the model metadata hash.
  # Raises ServiceUnavailableError if the service is down.
  def model_info
    response = HTTParty.get(
      "#{ML_SERVICE_URL}/models",
      timeout: CONNECT_TIMEOUT,
      headers: { "Accept" => "application/json" }
    )

    unless response.success?
      raise ServiceUnavailableError, "ML service returned HTTP #{response.code}"
    end

    JSON.parse(response.body)
  rescue HTTParty::Error, Errno::ECONNREFUSED, SocketError, Errno::EHOSTUNREACH => e
    raise ServiceUnavailableError, "Cannot reach ML service: #{e.message}"
  end

  # ----------------------------------------------------------------
  # Private helpers
  # ----------------------------------------------------------------
  private

  def build_body(title:, description:, rating:, review_count:, category:, price:)
    {
      title:        title.to_s.strip,
      description:  description.to_s.strip,
      rating:       rating.present?       ? rating.to_f       : 0,
      review_count: review_count.present? ? review_count.to_i : 0,
      category:     category.to_s.strip.presence || "unknown",
      price:        price.present?        ? price.to_f        : nil
    }.compact
  end

  def handle_response(response)
    case response.code
    when 200..299
      JSON.parse(response.body)
    when 503
      raise ServiceUnavailableError, "ML service returned 503 — model not loaded. Run the training pipeline."
    when 400
      parsed = safe_parse(response.body)
      raise PredictionError, parsed["error"] || "Bad request to ML service"
    when 500
      parsed = safe_parse(response.body)
      raise PredictionError, "ML service internal error: #{parsed['error'] || response.body.truncate(200)}"
    else
      raise PredictionError, "Unexpected response from ML service (HTTP #{response.code})"
    end
  end

  def safe_parse(body)
    JSON.parse(body)
  rescue JSON::ParserError
    { "error" => body.to_s.truncate(200) }
  end
end
