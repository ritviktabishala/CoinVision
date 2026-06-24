const API_BASE_URL = "http://localhost:8000/predictions";

window.ApiService = {
    async predictCoin(file) {
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Standard prediction call failed");
            }
            return await response.json();
        } catch (error) {
            console.error(error);
            throw error;
        }
    },

    async analyzeCoin(file) {
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Analysis computation failed");
            }
            return await response.json();
        } catch (error) {
            console.error(error);
            throw error;
        }
    }
};