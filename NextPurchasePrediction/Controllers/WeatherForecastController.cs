using Microsoft.AspNetCore.Mvc;
using NextPurchasePrediction.Services;

namespace NextPurchasePrediction.Controllers;

[ApiController]
[Route("[controller]")]
public class WeatherForecastController : ControllerBase
{
    private static readonly string[] Summaries = new[]
    {
        "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
    };

    private readonly ILogger<WeatherForecastController> _logger;
    private readonly IPredictionService _predictionService;

    public WeatherForecastController(ILogger<WeatherForecastController> logger, IPredictionService predictionService)
    {
        _logger = logger;
        _predictionService = predictionService;
    }

    [HttpGet("predict")]
    public async Task<IActionResult> Get()
    {
        var nonMl = await _predictionService.PredictPM();
        var mlResult = await _predictionService.PredictMl();
        return Ok(new
        {
            NonML = nonMl,
            MlResult = mlResult
        });
    }
    
    
}