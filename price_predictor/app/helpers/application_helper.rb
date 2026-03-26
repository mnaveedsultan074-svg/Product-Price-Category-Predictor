module ApplicationHelper
  # ----------------------------------------------------------------
  # Price-tier helpers
  # ----------------------------------------------------------------

  TIER_CONFIG = {
    "budget"    => { badge: "tier-budget",    hex: "#22c55e", label: "Budget",    icon: "" },
    "mid-range" => { badge: "tier-mid-range", hex: "#3b82f6", label: "Mid-Range", icon: "" },
    "premium"   => { badge: "tier-premium",   hex: "#f59e0b", label: "Premium",   icon: "" },
    "luxury"    => { badge: "tier-luxury",    hex: "#8b5cf6", label: "Luxury",    icon: "" },
  }.freeze

  # Return the CSS class string for a tier badge
  def tier_badge_class(tier)
    TIER_CONFIG.dig(tier.to_s.downcase, :badge) || "bg-gray-100 text-gray-700 border border-gray-200"
  end

  # Return the hex colour for a tier (for inline styles)
  def tier_hex(tier)
    TIER_CONFIG.dig(tier.to_s.downcase, :hex) || "#6b7280"
  end

  # Return a human-readable label for a tier
  def tier_label(tier)
    TIER_CONFIG.dig(tier.to_s.downcase, :label) || tier.to_s.capitalize
  end

  # Tailwind gradient classes per tier
  def tier_gradient_classes(tier)
    {
      "budget"    => "from-green-400 to-emerald-600",
      "mid-range" => "from-blue-400 to-indigo-600",
      "premium"   => "from-amber-400 to-orange-600",
      "luxury"    => "from-purple-400 to-pink-600",
    }.fetch(tier.to_s.downcase, "from-gray-400 to-gray-600")
  end

  # Tailwind text-colour class per tier
  def tier_text_class(tier)
    {
      "budget"    => "text-green-600",
      "mid-range" => "text-blue-600",
      "premium"   => "text-amber-600",
      "luxury"    => "text-purple-600",
    }.fetch(tier.to_s.downcase, "text-gray-600")
  end

  # Tailwind bar-fill class for confidence bars
  def tier_bar_class(tier)
    {
      "budget"    => "confidence-fill-budget",
      "mid-range" => "confidence-fill-mid-range",
      "premium"   => "confidence-fill-premium",
      "luxury"    => "confidence-fill-luxury",
    }.fetch(tier.to_s.downcase, "bg-gray-400")
  end

  # ----------------------------------------------------------------
  # Number / percentage helpers
  # ----------------------------------------------------------------

  # Format a float as a percentage with N decimal places
  def pct(value, decimals: 1)
    return "N/A" if value.nil?
    "#{value.to_f.round(decimals)}%"
  end

  # Format a float metric (0-1) as a percentage
  def metric_pct(value, decimals: 1)
    return "N/A" if value.nil?
    "#{(value.to_f * 100).round(decimals)}%"
  end

  # Colour class for an F1 metric value (0-1)
  def f1_colour_class(value)
    return "text-gray-400" if value.nil?
    v = value.to_f
    if v >= 0.75 then "text-green-600 font-bold"
    elsif v >= 0.60 then "text-amber-600 font-semibold"
    else "text-red-500 font-semibold"
    end
  end

  # ----------------------------------------------------------------
  # SHAP helpers
  # ----------------------------------------------------------------

  # Friendly name for a SHAP feature identifier
  def shap_feature_label(raw_name)
    return "Unknown" if raw_name.blank?
    name = raw_name.to_s
    case name
    when "rating"       then "Product Rating"
    when "log_review"   then "Review Count (log)"
    when "review_count" then "Review Count"
    when /^cat_(.+)/    then "Category: #{Regexp.last_match(1).tr('_', ' ').titleize}"
    when /^src_(.+)/    then "Source: #{Regexp.last_match(1).capitalize}"
    when /^feature_(\d+)$/ then "Text Feature ##{Regexp.last_match(1)}"
    else name.tr('_', ' ').truncate(30)
    end
  end

  # Width percentage for a SHAP bar (0-100), capped at 100
  def shap_bar_width(shap_value, max_abs)
    return 0 if max_abs.to_f.zero?
    [(shap_value.to_f.abs / max_abs.to_f * 100).round(1), 100].min
  end

  # CSS class for a SHAP bar (positive vs negative)
  def shap_bar_colour(shap_value)
    shap_value.to_f >= 0 ? "shap-bar-positive" : "shap-bar-negative"
  end

  # Text colour for a SHAP value
  def shap_value_colour(shap_value)
    shap_value.to_f >= 0 ? "text-indigo-600" : "text-rose-500"
  end

  # ----------------------------------------------------------------
  # Layout / UI helpers
  # ----------------------------------------------------------------

  # Page title helper - call set_page_title in controller/view
  def page_title(suffix = nil)
    base = "PriceTier AI"
    suffix.present? ? "#{suffix} | #{base}" : base
  end

  # Render a stat card (used in model_info)
  # Usage: stat_card("F1-Macro", "74.3%", colour_class: "text-green-600")
  def stat_card(label, value, colour_class: "text-gray-800", sub: nil)
    content_tag(:div, class: "bg-white rounded-2xl border border-gray-100 shadow-sm p-6") do
      concat content_tag(:p, label, class: "text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2")
      concat content_tag(:p, value, class: "text-4xl font-black #{colour_class}")
      concat content_tag(:p, sub, class: "text-xs text-gray-400 mt-2") if sub.present?
    end
  end
end
