using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;

namespace NextPurchasePrediction.Services;

public interface IPredictionService
{
    Task<string> PredictPM();
    Task<string> PredictMl();
}

public class PredictionService : IPredictionService
{
    private readonly MLContext _mlContext;
    private List<DateTime> _purchaseDates;
    public PredictionService()
    {
        _mlContext = new MLContext();
        DateTime startDate = new DateTime(2024, 1, 1);
        DateTime endDate = DateTime.Today;

        // Calculate the total number of days between the two dates
        int totalDays = (endDate - startDate).Days;
        Random random = new Random();
        _purchaseDates = Enumerable.Range(0, 100)
                                    .Select(_ => startDate.AddDays(random.Next(0, totalDays + 1)))
                                    .ToList();
    }
    public Task<string> PredictPM()
    { 
        // Convert dates to ordinal (numeric) format
        var ordinalDates = _purchaseDates.OrderBy(x => x).Select(date => date.ToOADate()).ToList();

        // Calculate differences between successive dates
        List<double> differences = new List<double>();
        for (int i = 1; i < ordinalDates.Count; i++)
        {
            differences.Add(ordinalDates[i] - ordinalDates[i - 1]);
        }

        // Calculate the average difference
        double averageDifference = differences.Average(); 
        // Predict the next purchase date
        double nextPurchaseOrdinal = ordinalDates.Last() + averageDifference;
        DateTime nextPurchaseDate = DateTime.FromOADate(nextPurchaseOrdinal);
        return Task.FromResult($"The next predicted purchase date is:  {nextPurchaseDate.ToString("yyyy-MM-dd")}");
    }


    public Task<string> PredictMl()
    {
        var data = _purchaseDates.OrderBy(x => x).Select(date => new PurchaseDataInput
        { Date = date, Value = (float)date.ToOADate() }).ToList();
        
        //Load Data
        IDataView dataview = _mlContext.Data.LoadFromEnumerable(data);
        
        // Define the forecasting pipeline
        var pipeline = _mlContext.Forecasting.ForecastBySsa(
            outputColumnName: nameof(PurchaseForecastOut.Forecast),
            inputColumnName: nameof(PurchaseDataInput.Value),
            windowSize: 3,
            seriesLength: data.Count,
            trainSize: data.Count,
            horizon: 1
        );
        
        // Train the model
        var model = pipeline.Fit(dataview);
        
        // Create a prediction engine
        var forecastEngine =  model.CreateTimeSeriesEngine<PurchaseDataInput, PurchaseForecastOut>(_mlContext);
        
        // Predict the next date
        var forecast = forecastEngine.Predict(); 
        // Convert the forecasted value back to DateTime
        DateTime nextPurchaseDate = DateTime.FromOADate(forecast.Forecast[0]);
        return Task.FromResult($"The next predicted purchase date is:  {nextPurchaseDate.ToString("yyyy-MM-dd")}");

    }

    
}


public class PurchaseDataInput
{
    public DateTime Date { get; set; }
    public float Value { get; set; }
}

public class PurchaseForecastOut
{
    public float[] Forecast { get; set; }
}