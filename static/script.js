function formatParameters(params) {
    if (!params || typeof params !== 'object' || Object.keys(params).length === 0) {
        return "—";
    }

    return Object.entries(params)
        .map(([key, value]) => `<strong>${key}:</strong> ${value}`)
        .join(', ');
}