#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get install libgomp1 -y

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["NextPurchasePrediction/NextPurchasePrediction.csproj", "NextPurchasePrediction/"]
RUN dotnet restore "NextPurchasePrediction/NextPurchasePrediction.csproj"
COPY . .
WORKDIR "/src/NextPurchasePrediction"
RUN dotnet build "NextPurchasePrediction.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "NextPurchasePrediction.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "NextPurchasePrediction.dll"]