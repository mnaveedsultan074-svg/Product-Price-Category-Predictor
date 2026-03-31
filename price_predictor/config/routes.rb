Rails.application.routes.draw do
  root "predictions#index"

  resources :predictions, only: [:index, :create] do
    collection do
      get :history
      get :model_info
    end
  end

  get "health", to: "predictions#health"

  # Built-in health check
  get "up" => "rails/health#show", as: :rails_health_check
end
