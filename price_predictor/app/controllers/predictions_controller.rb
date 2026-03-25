class PredictionsController < ApplicationController
  # GET /
  def index
    @result = nil
  end

  # POST /predictions
  def create
    pdata   = prediction_params
    service = MlPredictionService.new

    begin
      @result = service.predict(
        title:        pdata[:title],
        description:  pdata[:description],
        rating:       pdata[:rating],
        review_count: pdata[:review_count],
        category:     pdata[:category],
        price:        pdata[:price]
      )

      # Persist last 10 predictions in the session
      session[:predictions] ||= []
      session[:predictions].unshift(
        "title"      => pdata[:title].to_s.truncate(80),
        "prediction" => @result["prediction"],
        "confidence" => @result["confidence_pct"],
        "category"   => pdata[:category],
        "timestamp"  => Time.current.strftime("%d %b %Y %H:%M")
      )
      session[:predictions] = session[:predictions].first(10)

      render :show

    rescue MlPredictionService::ServiceUnavailableError => e
      flash.now[:error] = "ML service unavailable — #{e.message}. " \
                          "Please start the Python pipeline first (see setup instructions below)."
      render :index, status: :service_unavailable

    rescue MlPredictionService::PredictionError => e
      flash.now[:error] = "Prediction failed: #{e.message}"
      render :index, status: :unprocessable_entity

    rescue StandardError => e
      flash.now[:error] = "Unexpected error: #{e.message}"
      render :index, status: :internal_server_error
    end
  end

  # GET /predictions/history
  def history
    @predictions = session[:predictions] || []
  end

  # GET /predictions/model_info
  def model_info
    service = MlPredictionService.new
    @info   = service.model_info
  rescue MlPredictionService::ServiceUnavailableError => e
    @info = nil
    flash.now[:error] = "ML service is not running: #{e.message}"
  rescue StandardError => e
    @info = nil
    flash.now[:error] = "Could not fetch model info: #{e.message}"
  end

  # GET /health   (JSON endpoint used by monitoring / docker healthcheck)
  def health
    service     = MlPredictionService.new
    health_data = service.health_check
    render json: health_data
  rescue StandardError => e
    render json: { status: "error", message: e.message }, status: :service_unavailable
  end

  private

  def prediction_params
    params.require(:prediction)
          .permit(:title, :description, :rating, :review_count, :category, :price)
  end
end
